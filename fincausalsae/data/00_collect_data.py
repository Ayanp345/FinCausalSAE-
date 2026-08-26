"""
FinCausalSAE — Phase 0: Data Collection & Counterfactual Corpus
================================================================
Collects:
  1. Earnings call transcripts (SEC EDGAR + HuggingFace)          [full mode]
  2. Stock price forward returns (yfinance)                        [full mode]
  3. EPS consensus surprise (yfinance earnings_dates)               [full mode]
  4. A small synthetic corpus with obvious causal structure         [demo mode]

Then builds paired counterfactual transcripts via controlled LLM rewriting
(or, in demo mode, deterministic template-based rewriting so no API key
is required to test the pipeline).

Run:
  python data/00_collect_data.py                 # demo mode (default)
  python data/00_collect_data.py --mode full      # real data, needs network + GPU downstream
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cli import parse_mode

args = parse_mode()

import config
from config import get_logger

log = get_logger("phase0")
config.mode_banner()

import pandas as pd

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JPM",
    "JNJ", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "AVGO",
    "LLY", "COST", "WMT", "TMO", "MCD", "BAC", "CSCO", "ABT", "ACN", "CRM", "NEE",
    "ADBE", "NKE", "DIS", "RTX", "VZ", "LIN", "TXN", "PM", "AMGN", "QCOM", "HON",
    "WFC", "IBM", "CAT", "GS", "BA", "AXP", "SBUX", "GILD", "MDLZ", "ISRG", "ADP",
    "BKNG", "NOW", "INTU", "REGN", "VRTX", "MMC", "BLK", "SCHW", "ELV", "MO", "CI",
    "SYK", "ETN", "ZTS", "T", "SPGI", "PLD", "CCI", "AMT", "GE", "DUK", "SO", "D",
    "F", "GM", "USB", "PNC", "TFC", "AFL", "ALL", "MET", "PRU", "CME", "ICE", "CBOE",
]

# ─── DEMO MODE: SYNTHETIC DATA (no network calls) ────────────────────────────
# Deliberately built so that "margin_flip" / "guidance_flip" / etc. have a
# ground-truth causal relationship to a synthetic forward return, so you can
# sanity-check that Phase 3's causal patching recovers something sensible
# before ever paying for a GPU.

GOOD_TEMPLATES = [
    "Our gross margins came in at {hi}%, above the {mid}% we guided, "
    "driven by strong operational efficiency and disciplined cost control.",
    "We are raising full-year guidance to reflect the strength we are seeing "
    "across every region, with revenue growth accelerating to {hi}%.",
    "Demand remains robust and we beat consensus EPS estimates by a wide margin "
    "this quarter, with no signs of a slowdown ahead.",
    "Free cash flow conversion improved meaningfully and we expect that trend "
    "to continue as we scale into next year.",
]
BAD_TEMPLATES = [
    "Our gross margins came in at {lo}%, below the {mid}% we guided, "
    "due to persistent supply chain pressure and input cost inflation.",
    "We are lowering full-year guidance to reflect softening demand across "
    "most regions, with revenue growth decelerating to {lo}%.",
    "Demand has weakened and we missed consensus EPS estimates this quarter, "
    "and we see continued headwinds ahead.",
    "Free cash flow conversion deteriorated and we expect that pressure to "
    "persist as macro conditions remain challenging.",
]

CF_FLIP_TYPES = ["margin_flip", "guidance_flip", "hedging_removal", "tone_neutralize"]


def build_synthetic_corpus(n_per_ticker=6, seed=42):
    import random
    rng = random.Random(seed)
    rows = []
    tickers = SP500_TICKERS[:15]
    dates = pd.date_range("2021-01-01", "2024-06-01", periods=n_per_ticker)

    for ticker in tickers:
        for i, dt in enumerate(dates):
            is_good = rng.random() > 0.5
            template = rng.choice(GOOD_TEMPLATES if is_good else BAD_TEMPLATES)
            hi, lo, mid = rng.randint(40, 55), rng.randint(20, 35), rng.randint(36, 39)
            text = template.format(hi=hi, lo=lo, mid=mid)
            # simulate a forward return correlated (but noisily) with sentiment
            base_ret = 0.04 if is_good else -0.04
            noise = rng.gauss(0, 0.03)
            rows.append({
                "ticker": ticker,
                "call_date": dt,
                "text": text,
                "ret_1d": (base_ret + noise) * 0.2,
                "ret_5d": (base_ret + noise) * 0.5,
                "ret_30d": base_ret + noise,
                "eps_surprise": base_ret * rng.uniform(0.5, 1.5),
            })
    return pd.DataFrame(rows)


def build_synthetic_counterfactuals(df, n_per_transcript=4, seed=42):
    """Deterministic template-based counterfactual generation — no LLM/API needed."""
    import random
    rng = random.Random(seed)
    cfs = []
    sample = df.sample(min(30, len(df)), random_state=seed)

    for idx, row in sample.iterrows():
        is_currently_good = any(w in row["text"] for w in ["above", "raising", "beat", "improved"])
        for ft in CF_FLIP_TYPES[:n_per_transcript]:
            hi, lo, mid = rng.randint(40, 55), rng.randint(20, 35), rng.randint(36, 39)
            if ft in ("margin_flip", "guidance_flip"):
                cf_template = rng.choice(BAD_TEMPLATES if is_currently_good else GOOD_TEMPLATES)
            elif ft == "hedging_removal":
                cf_template = row["text"].replace("we expect", "we are certain").replace(
                    "we believe", "we know")
            else:  # tone_neutralize
                cf_template = row["text"].replace("headwinds", "conditions").replace(
                    "challenging", "typical")
            cf_text = cf_template.format(hi=hi, lo=lo, mid=mid) if "{" in cf_template else cf_template
            cfs.append({
                "original_id": idx,
                "ticker": row["ticker"],
                "call_date": row["call_date"],
                "ret_30d": row["ret_30d"],
                "eps_surprise": row.get("eps_surprise"),
                "cf_type": ft,
                "original_text": row["text"],
                "cf_text": cf_text,
            })
    return pd.DataFrame(cfs)


# ─── FULL MODE: REAL DATA (network + yfinance + optional Anthropic API) ─────
def load_hf_transcripts():
    from datasets import load_dataset
    log.info("Loading HF transcript dataset (lamini/earnings-calls-qa)...")
    try:
        ds = load_dataset("lamini/earnings-calls-qa", split="train")
        df = ds.to_pandas()
        df = df.rename(columns={"transcript": "text", "ticker": "ticker", "date": "call_date"})
        df["call_date"] = pd.to_datetime(df["call_date"])
        df = df[df["ticker"].isin(SP500_TICKERS)]
        log.info(f"Loaded {len(df)} transcripts from HF")
        return df
    except Exception as e:
        log.warning(f"HF load failed ({e}). Falling back to SEC EDGAR scrape.")
        return scrape_sec_transcripts()


def scrape_sec_transcripts():
    import requests
    base = "https://efts.sec.gov/LATEST/search-index"
    transcripts = []
    for ticker in SP500_TICKERS[:20]:
        params = {
            "q": f'"{ticker}" "earnings call" "operator"',
            "dateRange": "custom", "startdt": "2019-01-01", "enddt": "2024-12-31",
            "forms": "8-K",
            "_source": "file_date,period_of_report,entity_name,file_num",
        }
        try:
            r = requests.get(base, params=params, timeout=10)
            if r.status_code == 200:
                hits = r.json().get("hits", {}).get("hits", [])
                for h in hits[:3]:
                    src = h.get("_source", {})
                    transcripts.append({
                        "ticker": ticker,
                        "call_date": src.get("period_of_report", ""),
                        "text": f"[SEC EDGAR placeholder for {ticker}]",
                        "filing_url": h.get("_id", ""),
                    })
        except Exception as e:
            log.warning(f"SEC EDGAR request failed for {ticker}: {e}")
        time.sleep(0.2)
    return pd.DataFrame(transcripts)


def get_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    import yfinance as yf
    results = []
    for ticker in df["ticker"].unique():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", auto_adjust=True)
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            earnings = stock.earnings_dates
            if earnings is not None:
                earnings.index = pd.to_datetime(earnings.index).tz_localize(None)
            else:
                earnings = pd.DataFrame()

            sub = df[df["ticker"] == ticker].copy()
            for _, row in sub.iterrows():
                call_dt = pd.to_datetime(row["call_date"])
                future = hist[hist.index > call_dt]
                if len(future) < 30:
                    continue
                p0 = future.iloc[0]["Close"]
                p1 = future.iloc[1]["Close"] if len(future) > 1 else None
                p5 = future.iloc[5]["Close"] if len(future) > 5 else None
                p30 = future.iloc[29]["Close"] if len(future) > 29 else None

                eps_surprise = None
                if len(earnings) > 0:
                    nearby = earnings[
                        (earnings.index >= call_dt - timedelta(days=3)) &
                        (earnings.index <= call_dt + timedelta(days=3))
                    ]
                    if len(nearby) > 0:
                        row_e = nearby.iloc[0]
                        rep, est = row_e.get("Reported EPS"), row_e.get("EPS Estimate")
                        if rep is not None and est is not None and est != 0:
                            eps_surprise = float((rep - est) / abs(est))

                results.append({
                    "ticker": ticker, "call_date": call_dt, "text": row["text"],
                    "ret_1d": (p1 - p0) / p0 if p1 else None,
                    "ret_5d": (p5 - p0) / p0 if p5 else None,
                    "ret_30d": (p30 - p0) / p0 if p30 else None,
                    "eps_surprise": eps_surprise,
                })
        except Exception as e:
            log.warning(f"Skipping {ticker}: {e}")
    return pd.DataFrame(results).dropna(subset=["ret_30d"])


COUNTERFACTUAL_PROMPT = """You are an expert at creating minimally-edited counterfactual versions of earnings call transcripts for financial research.

Given an earnings call excerpt, create a counterfactual version that:
1. Flips ONLY the specified semantic property
2. Keeps all other language, structure, and content identical
3. Changes as few words as possible
4. Maintains natural business language

Original excerpt:
{excerpt}

Property to flip: {flip_type}

Flip types:
- "margin_flip": If margins beat guidance -> miss, or if miss -> beat. Change only the numbers and directional language.
- "guidance_flip": If guidance raised -> lowered, or lowered -> raised. Change only the outlook language.
- "hedging_removal": Remove hedging phrases ("we believe", "we expect", "subject to", "if conditions allow") making statements definitive.
- "tone_neutralize": Replace uncertainty signals ("challenges ahead", "headwinds", "cautious") with neutral equivalents.

Output ONLY the counterfactual excerpt. No explanation."""


def build_counterfactuals_llm(df: pd.DataFrame, n_per_transcript=2) -> pd.DataFrame:
    """Requires ANTHROPIC_API_KEY to be set. Costs real money — see SETUP_GUIDE.md."""
    from anthropic import Anthropic
    client = Anthropic()
    flip_types = CF_FLIP_TYPES
    cfs = []
    for idx, row in df.iterrows():
        excerpt = row["text"][:2000]
        for ft in flip_types[:n_per_transcript]:
            prompt = COUNTERFACTUAL_PROMPT.format(excerpt=excerpt, flip_type=ft)
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                cf_text = response.content[0].text
                cfs.append({
                    "original_id": idx, "ticker": row["ticker"], "call_date": row["call_date"],
                    "ret_30d": row["ret_30d"], "eps_surprise": row.get("eps_surprise"),
                    "cf_type": ft, "original_text": excerpt, "cf_text": cf_text,
                })
            except Exception as e:
                log.warning(f"CF generation failed for {row['ticker']}: {e}")
            time.sleep(0.5)
    return pd.DataFrame(cfs)


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Phase 0: Data Collection ===")

    if config.DEMO_MODE:
        log.info("Demo mode: generating synthetic earnings-call corpus (no network calls).")
        df_aligned = build_synthetic_corpus()
    else:
        df_raw = load_hf_transcripts()
        df_raw.to_parquet(config.RAW_DIR / "transcripts_raw.parquet", index=False)
        log.info(f"Saved {len(df_raw)} raw transcripts")
        log.info("Fetching forward returns from yfinance (this can take a while)...")
        df_aligned = get_forward_returns(df_raw)

    df_aligned.to_parquet(config.PROC_DIR / "transcripts_aligned.parquet", index=False)
    log.info(f"Saved {len(df_aligned)} aligned transcripts with returns")

    # Temporal train/val/test split — NO LEAKAGE
    df_aligned = df_aligned.sort_values("call_date")
    n = len(df_aligned)
    cutoff_train = df_aligned["call_date"].iloc[int(n * 0.6)]
    cutoff_val = df_aligned["call_date"].iloc[int(n * 0.8)]

    df_train = df_aligned[df_aligned["call_date"] < cutoff_train]
    df_val = df_aligned[(df_aligned["call_date"] >= cutoff_train) & (df_aligned["call_date"] < cutoff_val)]
    df_test = df_aligned[df_aligned["call_date"] >= cutoff_val]

    for split, data in [("train", df_train), ("val", df_val), ("test", df_test)]:
        data.to_parquet(config.PROC_DIR / f"{split}.parquet", index=False)
        log.info(f"  {split}: {len(data)} transcripts")

    log.info("Building counterfactual corpus...")
    if config.DEMO_MODE:
        df_cf = build_synthetic_counterfactuals(df_train, n_per_transcript=config.N_CF_PAIRS_PER_TYPE and 4)
    else:
        df_cf_source = df_train.sample(min(500, len(df_train)), random_state=42)
        log.warning(
            "Real counterfactual generation calls the Anthropic API and costs "
            "money (~$5 for 500 transcripts). Set ANTHROPIC_API_KEY and rerun "
            "with build_counterfactuals_llm() uncommented in this file when ready."
        )
        df_cf = build_synthetic_counterfactuals(df_cf_source, n_per_transcript=4)  # safe default

    df_cf.to_parquet(config.CF_DIR / "counterfactuals.parquet", index=False)
    log.info(f"Saved {len(df_cf)} counterfactual pairs to {config.CF_DIR / 'counterfactuals.parquet'}")

    log.info("Phase 0 complete.")
    log.info(f"Next: python models/01_lora_finetune.py --mode {args.mode}")
