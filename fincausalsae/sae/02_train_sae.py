"""
FinCausalSAE — Phase 2: Train Financial SAE
============================================
Trains a TopK Sparse Autoencoder on MLP layer outputs of the domain-adapted
model, so that Phase 3 can do causal patching in a sparse, interpretable
feature basis instead of raw (entangled) neuron space.

  DEMO mode : a small hand-written TopK SAE (pure PyTorch, no extra deps),
              trained on gpt2 activations from Phase 1's tiny corpus.
              Runs on CPU in well under a minute.
  FULL mode : SAELens's LanguageModelSAERunner — the standard research
              library — trained on Llama-3.1-8B activations. Needs a GPU.

Both modes save an SAE object exposing `.encode(x)`, `.decode(feats)` and
`.W_dec` (the [n_features, d_model] decoder matrix), so Phase 3's causal
patching code works unmodified against either one.

Key research idea (kept from the original design): we track how much each
feature's activation shifts between (real, counterfactual) transcript pairs
during/after training. Features with large shifts on a given cf_type are
strong *candidates* for causal circuits — Phase 3 then verifies which of
them are actually causal via activation patching (candidates != causal).

Run:
  python sae/02_train_sae.py                # demo
  python sae/02_train_sae.py --mode full     # real run, needs GPU
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cli import parse_mode

args = parse_mode()

import config
from config import get_logger

log = get_logger("phase2")
config.mode_banner()

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.simple_sae import SimpleTopKSAE


def train_demo_sae(model, tokenizer, texts):
    """Extracts layer activations with transformer_lens and trains SimpleTopKSAE."""
    log.info(f"Collecting activations from layer {config.TARGET_LAYER} on {len(texts)} texts...")
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"
    all_acts = []
    for text in tqdm(texts, desc="Collecting activations"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN)
        with torch.no_grad():
            _, cache = model.run_with_cache(enc["input_ids"], names_filter=hook_name)
        all_acts.append(cache[hook_name][0])  # (seq, d_model)
    acts = torch.cat(all_acts, dim=0)  # (total_tokens, d_model)
    log.info(f"Collected {acts.shape[0]} token activations of dim {acts.shape[1]}")

    sae = SimpleTopKSAE(config.D_MODEL, config.N_FEATURES, config.SAE_K)
    opt = torch.optim.Adam(sae.parameters(), lr=config.LR_SAE)

    n_steps = max(50, min(500, config.TRAINING_TOKS // max(len(acts), 1)))
    batch_size = min(64, len(acts))
    log.info(f"Training SAE for {n_steps} steps (batch size {batch_size})...")

    for step in range(n_steps):
        idx = torch.randint(0, len(acts), (batch_size,))
        batch = acts[idx]
        recon, feats = sae(batch)
        mse = ((recon - batch) ** 2).mean()
        l1 = feats.abs().mean()
        loss = mse + config.L1_COEFF * l1
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, n_steps // 10) == 0:
            log.info(f"  step {step:4d}/{n_steps} | mse={mse.item():.4f} | l1={l1.item():.4f}")

    return sae


# ─── SAELENS CONFIG (full mode, real research pipeline) ─────────────────────
def get_full_sae_config():
    from sae_lens import LanguageModelSAERunnerConfig
    return LanguageModelSAERunnerConfig(
        model_name=str(config.MODEL_OUT / "merged"),
        model_from_pretrained_kwargs={"dtype": torch.bfloat16},
        hook_name=f"blocks.{config.TARGET_LAYER}.hook_mlp_out",
        hook_layer=config.TARGET_LAYER,
        architecture="topk",
        d_in=config.D_MODEL,
        expansion_factor=config.EXPANSION,
        k=config.SAE_K,
        training_tokens=config.TRAINING_TOKS,
        l1_coefficient=config.L1_COEFF,
        lr=config.LR_SAE,
        train_batch_size_tokens=4096,
        context_size=config.MAX_SEQ_LEN,
        adam_beta1=0.9, adam_beta2=0.999,
        lr_scheduler_name="cosine_annealing_warmup",
        lr_warm_up_steps=1000, lr_decay_steps=5000,
        log_to_wandb=False,
        checkpoint_path=str(config.SAE_DIR),
        n_checkpoints=5,
        device="cuda", dtype="bfloat16",
        dataset_path=str(config.PROC_DIR / "train.parquet"),
        dataset_trust_remote_code=True,
        feature_sampling_window=1000,
        dead_feature_threshold=1e-6,
        dead_feature_window=5000,
        n_eval_batches=10,
    )


# ─── COUNTERFACTUAL-AWARE FEATURE DELTAS (both modes) ────────────────────────
def compute_cf_feature_deltas(model, sae, cf_df, tokenizer, device="cpu"):
    """
    For each counterfactual pair, computes per-feature activation delta.
    High-delta features for a given cf_type are CANDIDATES for a causal
    circuit — Phase 3 tests them with actual activation patching.
    """
    results = []
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"

    for _, row in tqdm(cf_df.iterrows(), total=len(cf_df), desc="CF deltas"):
        try:
            real_enc = tokenizer(row["original_text"], return_tensors="pt",
                                  truncation=True, max_length=config.MAX_SEQ_LEN)
            cf_enc = tokenizer(row["cf_text"], return_tensors="pt",
                                truncation=True, max_length=config.MAX_SEQ_LEN)

            with torch.no_grad():
                _, real_cache = model.run_with_cache(real_enc["input_ids"], names_filter=hook_name)
                _, cf_cache = model.run_with_cache(cf_enc["input_ids"], names_filter=hook_name)

            real_feats = sae.encode(real_cache[hook_name][0]).mean(0).detach().numpy()
            cf_feats = sae.encode(cf_cache[hook_name][0]).mean(0).detach().numpy()
            delta = np.abs(real_feats - cf_feats)
            results.append({"ticker": row["ticker"], "cf_type": row["cf_type"], "delta": delta})
        except Exception as e:
            log.warning(f"Error processing {row.get('ticker', '?')}: {e}")

    summary = []
    for cf_type in cf_df["cf_type"].unique():
        sub = [r["delta"] for r in results if r["cf_type"] == cf_type]
        if sub:
            mean_delta = np.stack(sub).mean(0)
            threshold = 0.001 if config.DEMO_MODE else 0.01
            for feat_id, d in enumerate(mean_delta):
                if d > threshold:
                    summary.append({"feature_id": feat_id, "cf_type": cf_type, "mean_delta": float(d)})
    return pd.DataFrame(summary)


def validate_sae(sae, model, tokenizer, val_texts, device="cpu"):
    """L0 sparsity + reconstruction cosine similarity sanity checks."""
    from torch.nn.functional import cosine_similarity
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"
    l0s, cos_sims = [], []

    for text in val_texts[:min(50, len(val_texts))]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN)
        with torch.no_grad():
            _, cache = model.run_with_cache(enc["input_ids"], names_filter=hook_name)
        hidden = cache[hook_name][0]
        feats = sae.encode(hidden)
        recon = sae.decode(feats)
        l0s.append((feats > 0).float().sum(-1).mean().item())
        cos_sims.append(cosine_similarity(hidden, recon, dim=-1).mean().item())

    metrics = {
        "l0_mean": float(np.mean(l0s)), "l0_std": float(np.std(l0s)),
        "cosine_sim_mean": float(np.mean(cos_sims)), "cosine_sim_std": float(np.std(cos_sims)),
    }
    log.info(f"SAE Validation: L0={metrics['l0_mean']:.1f}±{metrics['l0_std']:.1f} | "
              f"cos_sim={metrics['cosine_sim_mean']:.3f}±{metrics['cosine_sim_std']:.3f}")
    if metrics["cosine_sim_mean"] < 0.5:
        log.warning("Low reconstruction quality — SAE needs more training/data.")
    return metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Phase 2: SAE Training ===")

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    merged_dir = config.MODEL_OUT / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"{merged_dir} not found. Run `python models/01_lora_finetune.py "
            f"--mode {args.mode}` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading fine-tuned model into TransformerLens (HookedTransformer)...")
    model = HookedTransformer.from_pretrained(
        str(merged_dir), center_writing_weights=False,
        dtype=config.DTYPE, device=config.DEVICE,
    )
    model.eval()

    train_df = pd.read_parquet(config.PROC_DIR / "train.parquet")
    val_path = config.PROC_DIR / "val.parquet"
    val_texts = pd.read_parquet(val_path)["text"].tolist() if val_path.exists() else train_df["text"].tolist()[:10]

    if config.DEMO_MODE:
        sae = train_demo_sae(model, tokenizer, train_df["text"].tolist())
        torch.save(sae.state_dict(), config.SAE_DIR / "fin_sae_state.pt")
        # Save a small metadata file so Phase 3/4 can reconstruct the SAE shape.
        import json
        with open(config.SAE_DIR / "sae_meta.json", "w") as f:
            json.dump({"d_in": config.D_MODEL, "n_features": config.N_FEATURES, "k": config.SAE_K}, f)
        log.info(f"SAE saved to {config.SAE_DIR}/fin_sae_state.pt")
    else:
        from sae_lens import SAETrainingRunner
        cfg = get_full_sae_config()
        log.info(f"Training SAE on layer {config.TARGET_LAYER} "
                  f"(d={config.D_MODEL} -> {config.N_FEATURES} features)")
        runner = SAETrainingRunner(cfg)
        sae = runner.run()
        sae.save_model(str(config.SAE_DIR / "fin_sae_layer20"))
        log.info(f"SAE saved to {config.SAE_DIR}/fin_sae_layer20")

    validate_sae(sae, model, tokenizer, val_texts, device=config.DEVICE)

    cf_path = config.CF_DIR / "counterfactuals.parquet"
    if cf_path.exists():
        log.info("Computing counterfactual feature deltas (candidate causal features)...")
        cf_df = pd.read_parquet(cf_path)
        deltas = compute_cf_feature_deltas(model, sae, cf_df, tokenizer, device=config.DEVICE)
        deltas.to_parquet(config.SAE_DIR / "cf_feature_deltas.parquet", index=False)
        log.info(f"{len(deltas)} candidate causally-sensitive (feature, cf_type) pairs found")
    else:
        log.warning(f"No counterfactuals found at {cf_path} — run Phase 0 first.")

    log.info("Phase 2 complete.")
    log.info(f"Next: python circuits/03_causal_patching.py --mode {args.mode}")
