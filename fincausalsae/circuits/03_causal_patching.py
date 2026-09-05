import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cli import parse_mode

args = parse_mode()

import config
from config import get_logger

log = get_logger("phase3")
config.mode_banner()

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

POSITIVE_TOKENS = ["beat", "exceeded", "above", "raised", "strong", "growth"]
NEGATIVE_TOKENS = ["miss", "below", "lowered", "weak", "decline", "headwind"]


def get_sentiment_logit_diff(model, token_ids, tokenizer):
    """P(positive financial outcome) - P(negative financial outcome) at the last token."""
    pos_ids = [tokenizer.encode(" " + t, add_special_tokens=False)[0] for t in POSITIVE_TOKENS]
    neg_ids = [tokenizer.encode(" " + t, add_special_tokens=False)[0] for t in NEGATIVE_TOKENS]
    with torch.no_grad():
        logits = model(token_ids)
    last_logits = logits[0, -1, :]
    return (last_logits[pos_ids].mean() - last_logits[neg_ids].mean()).item()


def load_sae():
    if config.DEMO_MODE:
        from utils.simple_sae import load_demo_sae
        return load_demo_sae(config.SAE_DIR)
    else:
        from sae_lens import SAE
        sae, _, _ = SAE.from_pretrained(
            release="fin-sae-layer20", sae_id=str(config.SAE_DIR / "fin_sae_layer20"))
        return sae.to(config.DEVICE)


def scan_causal_features(model, sae, tokenizer, cf_df):
    """
    Strategy (computational efficiency):
      Step 1: Pre-filter using CF activation deltas from Phase 2 (cheap)
      Step 2: Full activation-patching experiment on the pre-filtered
              candidates only (expensive) -> real causal scores

    This is what turns an intractable "patch all N_FEATURES" scan into a
    tractable one, while still testing enough candidates to find the
    causal circuit.
    """
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"
    deltas_path = config.SAE_DIR / "cf_feature_deltas.parquet"
    cf_deltas = pd.read_parquet(deltas_path) if deltas_path.exists() else None
    if cf_deltas is None:
        log.warning("No CF deltas found — run Phase 2 first. Using random candidates.")

    causal_results = []
    top_k = config.PATCH_TOP_K_CANDIDATES
    per_pair_limit = min(top_k, 200 if config.DEMO_MODE else 200)

    for cf_type in cf_df["cf_type"].unique():
        log.info(f"Scanning causal features for cf_type='{cf_type}'...")
        subset = cf_df[cf_df["cf_type"] == cf_type].head(config.N_CF_PAIRS_PER_TYPE)

        if cf_deltas is not None and len(cf_deltas[cf_deltas["cf_type"] == cf_type]) > 0:
            candidates = (
                cf_deltas[cf_deltas["cf_type"] == cf_type]
                .nlargest(top_k, "mean_delta")["feature_id"].tolist()
            )
        else:
            candidates = np.random.choice(config.N_FEATURES, min(top_k, config.N_FEATURES), replace=False).tolist()

        feature_causal_scores = {fid: [] for fid in candidates}

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"  Patching {cf_type}"):
            real_enc = tokenizer(row["original_text"], return_tensors="pt",
                                  truncation=True, max_length=config.MAX_SEQ_LEN)
            cf_enc = tokenizer(row["cf_text"], return_tensors="pt",
                                truncation=True, max_length=config.MAX_SEQ_LEN)
            real_toks, cf_toks = real_enc["input_ids"], cf_enc["input_ids"]

            with torch.no_grad():
                _, real_cache = model.run_with_cache(real_toks, names_filter=hook_name)
                _, cf_cache = model.run_with_cache(cf_toks, names_filter=hook_name)

            real_hidden, cf_hidden = real_cache[hook_name][0], cf_cache[hook_name][0]
            real_feats, cf_feats = sae.encode(real_hidden), sae.encode(cf_hidden)
            baseline = get_sentiment_logit_diff(model, real_toks, tokenizer)

            for fid in candidates[:per_pair_limit]:
                delta_act = (cf_feats[:, fid] - real_feats[:, fid]).mean()
                if abs(delta_act.item()) < 0.001:
                    continue
                feat_dir = sae.W_dec[fid]

                def hook_fn(value, hook, _d=delta_act, _f=feat_dir):
                    value[0] += _d * _f.unsqueeze(0)
                    return value

                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    with torch.no_grad():
                        patched = get_sentiment_logit_diff(model, real_toks, tokenizer)

                feature_causal_scores[fid].append(patched - baseline)

        for fid, scores in feature_causal_scores.items():
            if scores:
                causal_results.append({
                    "feature_id": fid, "cf_type": cf_type,
                    "causal_mean": float(np.mean(scores)),
                    "causal_abs_mean": float(np.mean(np.abs(scores))),
                    "causal_std": float(np.std(scores)),
                    "n_pairs": len(scores),
                })

    return pd.DataFrame(causal_results)


def auto_interpret_features(causal_df, corpus_texts, model, sae, tokenizer, top_n=None):
    """
    For each top causal feature, finds the text spans that activate it most
    strongly, then asks Claude (via the Anthropic API) for a short label.

    Requires ANTHROPIC_API_KEY. If it isn't set, falls back to a purely
    mechanical label ("Feature_<id>") so the pipeline still completes —
    interpretation quality is a nice-to-have, not a blocker.
    """
    import os
    top_n = top_n or config.N_TOP_CAUSAL
    top_features = causal_df.nlargest(top_n, "causal_abs_mean")["feature_id"].tolist()
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"

    have_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    client = None
    if have_api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic()
        except Exception as e:
            log.warning(f"Anthropic client unavailable ({e}); using mechanical labels.")

    feature_library = {}
    n_texts_to_scan = min(30 if config.DEMO_MODE else 500, len(corpus_texts))

    for fid in tqdm(top_features, desc="Interpreting features"):
        top_examples = []
        for text in corpus_texts[:n_texts_to_scan]:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN)
            with torch.no_grad():
                _, cache = model.run_with_cache(enc["input_ids"], names_filter=hook_name)
            feats = sae.encode(cache[hook_name][0])
            acts = feats[:, fid].detach().numpy()
            if acts.max() <= 0:
                continue
            for pos in np.where(acts > acts.max() * 0.8)[0]:
                start, end = max(0, pos - 15), min(len(enc["input_ids"][0]), pos + 15)
                span = tokenizer.decode(enc["input_ids"][0][start:end])
                top_examples.append((float(acts[pos]), span))

        top_examples = sorted(top_examples, key=lambda x: -x[0])[:20]
        example_texts = [e[1] for e in top_examples]

        label = f"Feature_{fid}"
        if client is not None and example_texts:
            prompt = (
                "These text excerpts all strongly activate the same internal "
                "feature in a financial language model. What financial concept "
                "does this feature detect? Answer in 5 words or less.\n\nExcerpts:\n"
                + "\n".join(f'- "{e}"' for e in example_texts[:10]) + "\n\nLabel:"
            )
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
                label = resp.content[0].text.strip()
            except Exception as e:
                log.warning(f"Auto-label failed for feature {fid}: {e}")

        feature_library[str(fid)] = {"label": label, "examples": example_texts}

    return feature_library


if __name__ == "__main__":
    log.info("=== Phase 3: Causal Circuit Discovery ===")

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    merged_dir = config.MODEL_OUT / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"{merged_dir} not found. Run Phase 1 first: "
            f"python models/01_lora_finetune.py --mode {args.mode}"
        )

    log.info("Loading model and SAE...")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained(
        str(merged_dir), center_writing_weights=False,
        dtype=config.DTYPE, device=config.DEVICE,
    )
    model.eval()
    sae = load_sae()

    cf_path = config.CF_DIR / "counterfactuals.parquet"
    if not cf_path.exists():
        raise FileNotFoundError(
            f"{cf_path} not found. Run Phase 0 first: "
            f"python data/00_collect_data.py --mode {args.mode}"
        )
    cf_df = pd.read_parquet(cf_path)
    log.info(f"Loaded {len(cf_df)} counterfactual pairs")

    log.info("Running causal feature scan (this is the novel contribution)...")
    causal_df = scan_causal_features(model, sae, tokenizer, cf_df)
    causal_df.to_parquet(config.CIRCUIT_DIR / "causal_scores.parquet", index=False)
    log.info(f"Causal scores computed for {len(causal_df)} (feature, cf_type) pairs")

    if len(causal_df) == 0:
        log.warning(
            "No causal features found. In demo mode this can happen with very "
            "little data — try increasing N_CF_PAIRS_PER_TYPE in config.py, or "
            "just proceed; Phase 4 will fall back gracefully."
        )
    else:
        log.info("Top causal features by absolute causal effect:")
        print(causal_df.nlargest(min(20, len(causal_df)), "causal_abs_mean")[
            ["feature_id", "cf_type", "causal_abs_mean", "causal_mean"]
        ].to_string(index=False))

        log.info("Auto-interpreting top causal features...")
        corpus_texts = pd.read_parquet(config.PROC_DIR / "train.parquet")["text"].tolist()
        feature_library = auto_interpret_features(causal_df, corpus_texts, model, sae, tokenizer)
        with open(config.CIRCUIT_DIR / "causal_feature_library.json", "w") as f:
            json.dump(feature_library, f, indent=2)

        print("\n=== CAUSAL FINANCIAL FEATURE LIBRARY ===")
        for _, row in causal_df.nlargest(min(20, len(causal_df)), "causal_abs_mean").iterrows():
            fid = int(row["feature_id"])
            label = feature_library.get(str(fid), {}).get("label", "?")
            print(f"  Feature #{fid:6d} | {row['cf_type']:20s} | "
                  f"causal={row['causal_abs_mean']:.4f} | {label}")

    log.info(f"Phase 3 complete. Results in {config.CIRCUIT_DIR}/")
    log.info(f"Next: python backtest/04_portfolio_backtest.py --mode {args.mode}")
