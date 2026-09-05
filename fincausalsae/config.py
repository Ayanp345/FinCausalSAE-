import os
import logging

import torch


DEMO_MODE = os.environ.get("FINCAUSAL_DEMO", "1") != "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name):
    return logging.getLogger(name)


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # Apple Silicon
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if DEMO_MODE and DEVICE != "cpu":
    # Demo mode is designed to also work fine on a GPU (it'll just be fast).
    pass
if not DEMO_MODE and DEVICE == "cpu":
    logging.getLogger("config").warning(
        "FULL mode selected but no GPU detected — this WILL be extremely "
        "slow or run out of memory loading an 8B-parameter model. See "
        "SETUP_GUIDE.md for free/cheap cloud GPU options."
    )

if DEMO_MODE:
    BASE_MODEL     = "gpt2"                 # 124M params, downloads in seconds
    D_MODEL        = 768
    N_LAYERS       = 12
    TARGET_LAYER   = 6                      # mid-layer, analogous role to L20/32 in Llama-8B
    EXPANSION      = 8
    N_FEATURES     = D_MODEL * EXPANSION    # 6,144 SAE features
    SAE_K          = 16                     # active features per token
    TRAINING_TOKS  = 200_000                # 200K tokens (~seconds on CPU)
    MAX_SEQ_LEN    = 256
    BATCH_SIZE     = 2
    GRAD_ACCUM     = 2
    LORA_R         = 8
    LORA_ALPHA     = 16
    EPOCHS         = 1
    DTYPE          = torch.float32          # bf16 often unsupported on CPU
    N_TOP_CAUSAL   = 15
    N_TOP_CORR     = 15
    PATCH_TOP_K_CANDIDATES = 200            # feature pre-filter size for patching scan
    N_CF_PAIRS_PER_TYPE = 5
else:
    BASE_MODEL     = "meta-llama/Llama-3.1-8B"   # BASE, not Instruct
    D_MODEL        = 4096
    N_LAYERS       = 32
    TARGET_LAYER   = 20
    EXPANSION      = 32
    N_FEATURES     = D_MODEL * EXPANSION    # 131,072 SAE features
    SAE_K          = 50
    TRAINING_TOKS  = 200_000_000
    MAX_SEQ_LEN    = 512
    BATCH_SIZE     = 4
    GRAD_ACCUM     = 8
    LORA_R         = 16
    LORA_ALPHA     = 32
    EPOCHS         = 2
    DTYPE          = torch.bfloat16
    N_TOP_CAUSAL   = 50
    N_TOP_CORR     = 50
    PATCH_TOP_K_CANDIDATES = 5000
    N_CF_PAIRS_PER_TYPE = 50

LORA_DROPOUT = 0.05
LR_LORA      = 2e-4
LR_SAE       = 4e-4
L1_COEFF     = 8e-5

from pathlib import Path

ROOT_DIR     = Path(__file__).resolve().parent
RAW_DIR      = ROOT_DIR / "data" / "raw"
PROC_DIR     = ROOT_DIR / "data" / "processed"
CF_DIR       = ROOT_DIR / "data" / "counterfactuals"
MODEL_OUT    = ROOT_DIR / "models" / "fin-lora"
SAE_DIR      = ROOT_DIR / "sae" / "checkpoints"
CIRCUIT_DIR  = ROOT_DIR / "circuits" / "results"
BACKTEST_DIR = ROOT_DIR / "backtest" / "results"

for d in [RAW_DIR, PROC_DIR, CF_DIR, MODEL_OUT, SAE_DIR, CIRCUIT_DIR, BACKTEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def mode_banner():
    """Print a loud, unmissable banner telling the user which mode is active."""
    label = "DEMO (GPT-2, synthetic data)" if DEMO_MODE else "FULL (Llama-3.1-8B, real data)"
    bar = "=" * 70
    print(bar)
    print(f" FinCausalSAE — MODE: {label}")
    print(f" Device: {DEVICE} | Model: {BASE_MODEL} | Target layer: {TARGET_LAYER}")
    print(f" SAE features: {N_FEATURES:,} | Training tokens: {TRAINING_TOKS:,}")
    print(bar)
