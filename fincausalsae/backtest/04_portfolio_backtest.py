
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cli import parse_mode

args = parse_mode()

import config
from config import get_logger

log = get_logger("phase4")
config.mode_banner()

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUINTILE = 5


def load_sae():
    if config.DEMO_MODE:
        from utils.simple_sae import load_demo_sae
        return load_demo_sae(config.SAE_DIR)
    else:
        from sae_lens import SAE
        sae, _, _ = SAE.from_pretrained(
            release="fin-sae-layer20", sae_id=str(config.SAE_DIR / "fin_sae_layer20"))
        return sae.to(config.DEVICE)


def extract_feature_activations(model, sae, tokenizer, df, feature_ids):
    hook_name = f"blocks.{config.TARGET_LAYER}.hook_mlp_out"
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting activations"):
        enc = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN)
        with torch.no_grad():
            _, cache = model.run_with_cache(enc["input_ids"], names_filter=hook_name)
        feats = sae.encode(cache[hook_name][0]).mean(0).detach().numpy()
        results.append({
            "ticker": row["ticker"], "call_date": row["call_date"],
            "ret_1d": row["ret_1d"], "ret_5d": row["ret_5d"], "ret_30d": row["ret_30d"],
            "eps_surprise": row.get("eps_surprise", 0),
            **{f"feat_{fid}": feats[fid] for fid in feature_ids},
        })
    return pd.DataFrame(results)


def build_causal_signal(feat_df, causal_scores_df, top_n):
    if causal_scores_df is None or len(causal_scores_df) == 0:
        return pd.Series(0.0, index=feat_df.index)
    causal_top = causal_scores_df.nlargest(min(top_n, len(causal_scores_df)), "causal_abs_mean")[
        ["feature_id", "causal_mean"]]
    signal = pd.Series(0.0, index=feat_df.index)
    for _, row in causal_top.iterrows():
        col = f"feat_{int(row['feature_id'])}"
        if col in feat_df.columns:
            signal += feat_df[col] * row["causal_mean"]
    return signal


def build_correlational_signal(feat_df, top_n):
    feat_cols = [c for c in feat_df.columns if c.startswith("feat_")]
    if not feat_cols:
        return pd.Series(0.0, index=feat_df.index)
    corrs = feat_df[feat_cols + ["ret_30d"]].corr()["ret_30d"].drop("ret_30d").fillna(0)
    top_feats = corrs.abs().nlargest(min(top_n, len(corrs)))
    signal = pd.Series(0.0, index=feat_df.index)
    for col, corr in top_feats.items():
        signal += feat_df[col] * np.sign(corrs[col])
    return signal


LEXICON_POS = {"beat", "exceeded", "above", "raised", "strong", "growth", "improved", "robust"}
LEXICON_NEG = {"miss", "missed", "below", "lowered", "weak", "decline", "headwind", "pressure",
               "deteriorated", "softening", "challenging"}


def lexicon_sentiment(text):
    words = text.lower().replace(",", "").replace(".", "").split()
    pos = sum(w in LEXICON_POS for w in words)
    neg = sum(w in LEXICON_NEG for w in words)
    return float(pos - neg)


def build_finbert_signal(texts):
    """Tries FinBERT; falls back to a simple lexicon score if unavailable offline."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        log.info("Loading FinBERT (ProsusAI/finbert)...")
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()
        scores = []
        for text in tqdm(texts, desc="FinBERT"):
            enc = tok(text[:512], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**enc).logits[0]
            scores.append((logits[0] - logits[1]).item())  # positive - negative
        return pd.Series(scores)
    except Exception as e:
        log.warning(f"FinBERT unavailable ({e}); falling back to lexicon-based sentiment baseline.")
        return pd.Series([lexicon_sentiment(t) for t in texts])


def quintile_backtest(df, signal, return_col="ret_30d", n_quintiles=QUINTILE):
    df = df.copy()
    df["signal"] = signal.values
    df["ym"] = pd.to_datetime(df["call_date"]).dt.to_period("M")

    monthly_returns = []
    for period, group in df.groupby("ym"):
        if len(group) < max(2, n_quintiles):
            continue
        group = group.sort_values("signal")
        q_size = max(1, len(group) // n_quintiles)
        short_leg = group.head(q_size)[return_col].mean()
        long_leg = group.tail(q_size)[return_col].mean()
        monthly_returns.append({
            "period": str(period), "long_return": long_leg, "short_return": short_leg,
            "ls_return": long_leg - short_leg, "n_stocks": len(group),
        })

    ret_df = pd.DataFrame(monthly_returns)
    if len(ret_df) == 0:
        return ret_df, {}

    ls_rets = ret_df["ls_return"]
    summary = {
        "mean_monthly_return": float(ls_rets.mean()),
        "std_monthly_return": float(ls_rets.std()),
        "sharpe_ratio": float(ls_rets.mean() / ls_rets.std() * np.sqrt(12)) if ls_rets.std() > 0 else 0.0,
        "win_rate": float((ls_rets > 0).mean()),
        "max_drawdown": float((ls_rets.cumsum() - ls_rets.cumsum().cummax()).min()),
        "annualized_return": float(ls_rets.mean() * 12),
        "n_periods": int(len(ret_df)),
    }
    return ret_df, summary


def compare_portfolios(causal_ret, corr_ret, finbert_ret, results):
    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Causal SAE':>12} {'Corr SAE':>12} {'FinBERT':>12}")
    print("-" * 60)
    metrics = [
        ("Sharpe Ratio (annualized)", "sharpe_ratio"),
        ("Mean Monthly Return", "mean_monthly_return"),
        ("Annualized Return", "annualized_return"),
        ("Win Rate", "win_rate"),
        ("Max Drawdown", "max_drawdown"),
        ("N Periods", "n_periods"),
    ]
    for label, key in metrics:
        c = results.get("causal", {}).get(key, 0)
        r = results.get("corr", {}).get(key, 0)
        f = results.get("finbert", {}).get(key, 0)
        print(f"{label:<30} {c:>12.4f} {r:>12.4f} {f:>12.4f}")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for ret_df, label, ls in [(causal_ret, "Causal SAE (Ours)", "-"),
                                (corr_ret, "Correlational SAE (SAE-FiRE)", "--"),
                                (finbert_ret, "FinBERT / Lexicon Sentiment", ":")]:
        if ret_df is not None and len(ret_df) > 0:
            ax.plot(ret_df["ls_return"].cumsum().values, label=label, lw=2, ls=ls)
    ax.set_title("Cumulative L/S Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)

    ax = axes[1]
    labels = ["Causal SAE\n(Ours)", "Corr SAE\n(SAE-FiRE)", "FinBERT"]
    sharpes = [results.get(k, {}).get("sharpe_ratio", 0) for k in ("causal", "corr", "finbert")]
    bars = ax.bar(labels, sharpes, color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.8, edgecolor="black")
    ax.set_title("Annualized Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="black", lw=0.5)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}",
                 ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = config.BACKTEST_DIR / "portfolio_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to {out_path}")


if __name__ == "__main__":
    log.info("=== Phase 4: Portfolio Backtest ===")

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    merged_dir = config.MODEL_OUT / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"{merged_dir} not found. Run Phase 1 first.")

    df_test_path = config.PROC_DIR / "test.parquet"
    df_train_path = config.PROC_DIR / "train.parquet"
    if not df_test_path.exists():
        raise FileNotFoundError(f"{df_test_path} not found. Run Phase 0 first.")

    df_test = pd.read_parquet(df_test_path)
    df_train = pd.read_parquet(df_train_path)
    if len(df_test) == 0:
        log.warning("Test split is empty (tiny demo dataset) — using train split for a smoke test instead.")
        df_test = df_train.copy()

    causal_scores_path = config.CIRCUIT_DIR / "causal_scores.parquet"
    causal_scores = pd.read_parquet(causal_scores_path) if causal_scores_path.exists() else None
    top_causal_ids = (
        causal_scores.nlargest(min(config.N_TOP_CAUSAL, len(causal_scores)), "causal_abs_mean")
        ["feature_id"].astype(int).tolist()
        if causal_scores is not None and len(causal_scores) > 0 else
        list(range(min(config.N_TOP_CAUSAL, config.N_FEATURES)))
    )

    log.info("Loading model and SAE for inference...")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained(
        str(merged_dir), center_writing_weights=False, dtype=config.DTYPE, device=config.DEVICE)
    model.eval()
    sae = load_sae()

    log.info("Extracting features on test set...")
    feat_test = extract_feature_activations(model, sae, tokenizer, df_test, top_causal_ids)

    log.info("Building portfolio signals...")
    feat_test["signal_causal"] = build_causal_signal(feat_test, causal_scores, config.N_TOP_CAUSAL)
    feat_test["signal_corr"] = build_correlational_signal(feat_test, config.N_TOP_CORR)
    feat_test["signal_finbert"] = build_finbert_signal(df_test["text"].tolist())

    log.info("Running backtests...")
    causal_ret, causal_summary = quintile_backtest(feat_test, feat_test["signal_causal"])
    corr_ret, corr_summary = quintile_backtest(feat_test, feat_test["signal_corr"])
    finbert_ret, finbert_summary = quintile_backtest(feat_test, feat_test["signal_finbert"])

    all_results = {"causal": causal_summary, "corr": corr_summary, "finbert": finbert_summary}

    with open(config.BACKTEST_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    if len(causal_ret):
        causal_ret.to_parquet(config.BACKTEST_DIR / "causal_returns.parquet", index=False)
    if len(corr_ret):
        corr_ret.to_parquet(config.BACKTEST_DIR / "corr_returns.parquet", index=False)

    compare_portfolios(causal_ret, corr_ret, finbert_ret, all_results)

    log.info("Phase 4 complete. All FinCausalSAE phases finished.")
    log.info(f"Results directory: {config.BACKTEST_DIR}")
    if config.DEMO_MODE:
        log.info(
            "This was a DEMO run (tiny synthetic data, gpt2). Numbers are NOT "
            "meaningful research results — the goal was only to prove the "
            "pipeline runs end-to-end. Rerun with --mode full on a GPU for "
            "real results. See SETUP_GUIDE.md."
        )
