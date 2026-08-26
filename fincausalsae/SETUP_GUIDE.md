# FinCausalSAE — Setup & Run Guide (No GPU Required To Start)

This guide is written for someone with **no local GPU**. It has two parts:

1. **Part A — Demo mode.** Run the *entire* pipeline on your own laptop's
   CPU in a few minutes, using GPT-2 and a small synthetic dataset. This
   proves your code, environment, and logic all work before you spend any
   money on cloud compute.
2. **Part B — Full mode.** Once the demo works, rent a cloud GPU for a few
   hours to run the real pipeline (Llama-3.1-8B, real market data, the full
   131K-feature SAE) and get research-grade results.

Everything in this repo is written so that Part A and Part B run the exact
same code — only the scale of the model/data/SAE changes (`config.py`
handles this via a `--mode demo|full` flag).

---

## Part A — Run the demo locally (CPU only, ~10 minutes, $0)

### A.1 Install Python

You need Python 3.10+ (3.9 mostly works too). Check with:

```bash
python3 --version
```

If you don't have it: [python.org/downloads](https://www.python.org/downloads/)
(macOS/Windows) or `sudo apt install python3.11 python3.11-venv` (Ubuntu/Debian).

### A.2 Unzip and set up a virtual environment

```bash
unzip fincausalsae.zip
cd fincausalsae

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-demo.txt
```

This installs only the lightweight demo dependencies (~1-2 GB, mostly
PyTorch CPU + Transformers). It does **not** install `bitsandbytes`,
`sae-lens`, or `yfinance` — those are only needed for full mode.

### A.3 Run the whole pipeline in one command

```bash
bash run_demo.sh
```

Or run each phase individually so you can inspect intermediate outputs:

```bash
python3 data/00_collect_data.py            # generates synthetic earnings-call data
python3 models/01_lora_finetune.py         # LoRA-tunes gpt2 on it (CPU, ~1-2 min)
python3 sae/02_train_sae.py                # trains a small SAE on gpt2 activations
python3 circuits/03_causal_patching.py     # THE CORE STEP: causal feature discovery
python3 backtest/04_portfolio_backtest.py  # compares causal vs correlational signals
```

Every script defaults to `--mode demo`, so you never have to pass a flag
during this part.

### A.4 What you should see

- `data/processed/{train,val,test}.parquet` — synthetic earnings-call rows
- `data/counterfactuals/counterfactuals.parquet` — paired counterfactual text
- `models/fin-lora/merged/` — the LoRA-tuned gpt2 checkpoint
- `sae/checkpoints/fin_sae_state.pt` — the trained SAE weights
- `circuits/results/causal_scores.parquet` and `causal_feature_library.json`
- `backtest/results/summary.json` and `portfolio_comparison.png`

**Important:** demo-mode numbers (Sharpe ratios, causal scores, etc.) are
**not meaningful research results** — the dataset is 90 synthetic rows and
the model is 124M parameters. The point of Part A is purely to confirm
the *pipeline* — every function, every file format, every hand-off between
phases — works before you spend money on a GPU.

### A.5 Troubleshooting Part A

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch` | Re-run `pip install -r requirements-demo.txt` inside the activated venv |
| Very slow / hangs on first run | The first run downloads gpt2 (~500MB) from Hugging Face — check your internet connection |
| `OSError: ... gpt2 ...` | You're offline; gpt2 must be downloaded once and is then cached in `~/.cache/huggingface` |
| Out of memory | Shouldn't happen with gpt2 on CPU; if it does, lower `BATCH_SIZE` in `config.py`'s demo branch |

---

## Part B — Run the real pipeline on a rented GPU

The full pipeline fine-tunes **Llama-3.1-8B**, trains a **131,072-feature
SAE**, and runs thousands of activation-patching forward passes. This
needs a GPU with **at least 24GB VRAM** (ideally 40-80GB for comfortable
batch sizes). Below are your options, cheapest/easiest first.

### Option 1 — Google Colab (easiest, free tier available)

**Free tier:** one T4 GPU (16GB VRAM), session limits (~12h, can disconnect
if idle). Good enough for Phase 1 (LoRA) with small batch size and
gradient checkpointing, and for Phase 3/4 with a reduced feature-candidate
count. Phase 2 (full 131K-feature SAE, 200M tokens) is slow on a T4 —
consider Colab Pro for this phase.

**Colab Pro ($9.99/mo)** or **Pro+ ($49.99/mo)**: access to A100/L4 GPUs
and longer runtimes — much closer to the original research budget.

Steps:
1. Go to [colab.research.google.com](https://colab.research.google.com), create a new notebook.
2. `Runtime → Change runtime type → GPU` (pick A100 if you have Pro).
3. Upload `fincausalsae.zip` (or `git clone` if you push this repo to
   GitHub first) and unzip:
   ```python
   from google.colab import files
   files.upload()   # select fincausalsae.zip
   !unzip -q fincausalsae.zip
   %cd fincausalsae
   ```
4. Install full dependencies:
   ```python
   !pip install -q -r requirements.txt
   ```
5. Set your Hugging Face token (needed for the gated `meta-llama/Llama-3.1-8B`
   weights — request access on the model page first) and Anthropic key if
   you want LLM-generated counterfactuals:
   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_..."
   os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # optional
   !huggingface-cli login --token $HF_TOKEN
   ```
6. Run phases one cell at a time so you can catch disconnects early:
   ```python
   !python data/00_collect_data.py --mode full
   !python models/01_lora_finetune.py --mode full
   !python sae/02_train_sae.py --mode full
   !python circuits/03_causal_patching.py --mode full
   !python backtest/04_portfolio_backtest.py --mode full
   ```
7. **Save outputs before your session ends** — Colab disks are ephemeral:
   ```python
   from google.colab import files
   !zip -r results.zip circuits/results backtest/results sae/checkpoints
   files.download("results.zip")
   ```
   Better: mount Google Drive at the start (`from google.colab import drive;
   drive.mount('/content/drive')`) and point `config.py`'s `ROOT_DIR`-derived
   paths at a Drive folder so checkpoints survive disconnects automatically.

### Option 2 — Kaggle Notebooks (free, no credit card)

Kaggle gives every account ~30 GPU-hours/week free (2x T4 or a P100),
no credit card required. Good for Phase 1 and Phase 3/4; Phase 2's full
SAE training may need to run in a couple of weekly sessions or be
scaled down (lower `EXPANSION` in `config.py`'s full branch).

Steps:
1. [kaggle.com](https://kaggle.com) → Create → New Notebook.
2. Settings (right sidebar) → Accelerator → GPU T4 x2 (or P100).
3. Upload the repo as a Kaggle Dataset, or `!git clone` your GitHub repo.
4. Same install/run commands as the Colab steps above.
5. Kaggle sessions persist output in `/kaggle/working` until you commit
   the notebook — commit regularly so results aren't lost.

### Option 3 — Rent a GPU by the hour (RunPod, Lambda Labs, Vast.ai)

This matches the original research budget in the README (~$94 total,
~47 GPU-hours on an A100). Recommended if you want a completely
uninterrupted run without session-timeout risk.

**RunPod** (used in the original design):
1. Create an account at [runpod.io](https://runpod.io) and add credit.
2. `Deploy → GPU Pod`, choose a template like "RunPod PyTorch 2.x", pick
   an **A100 80GB SXM** (or A6000 48GB if 80GB isn't available/needed),
   set **Secure Cloud** for reliability. On-demand A100 SXM runs roughly
   $2-2.5/hr — check current pricing on the site, it changes.
3. Once the pod is running, open its **Web Terminal** or connect via SSH.
4. Inside the pod:
   ```bash
   git clone <your-repo-url> fincausalsae   # or scp/upload the zip
   cd fincausalsae
   pip install -r requirements.txt
   export HF_TOKEN=hf_...
   export ANTHROPIC_API_KEY=sk-ant-...      # optional
   huggingface-cli login --token $HF_TOKEN

   python data/00_collect_data.py --mode full
   python models/01_lora_finetune.py --mode full
   python sae/02_train_sae.py --mode full
   python circuits/03_causal_patching.py --mode full
   python backtest/04_portfolio_backtest.py --mode full
   ```
5. Use `tmux` or `screen` so long-running phases (especially Phase 2's SAE
   training, ~20-30 GPU-hours) survive SSH disconnects:
   ```bash
   tmux new -s fincausal
   # ... run your commands ...
   # Ctrl+B then D to detach; `tmux attach -t fincausal` to reattach
   ```
6. **Stop or terminate the pod as soon as you're done** — you're billed
   by the minute. Download results first:
   ```bash
   # from your local machine
   scp -r root@<pod-ip>:/workspace/fincausalsae/circuits/results ./
   scp -r root@<pod-ip>:/workspace/fincausalsae/backtest/results ./
   ```

**Lambda Labs** ([lambdalabs.com](https://lambdalabs.com)) and
**Vast.ai** ([vast.ai](https://vast.ai)) work the same way — rent an
instance with an A100/A6000/4090, SSH in, and run the same commands.
Vast.ai is typically the cheapest (community/spot pricing) but less
reliable; Lambda Labs is closer to on-demand cloud pricing with good
availability.

### Option 4 — Reduce the GPU requirement instead of renting a bigger one

If you only have access to a smaller GPU (e.g. a single RTX 3090/4090
with 24GB), you can still run full mode with some knobs turned down in
`config.py`'s `else:` (full-mode) branch:

- Lower `EXPANSION` from 32 to 8 or 16 (fewer SAE features)
- Lower `TRAINING_TOKS` (fewer SAE training steps)
- Lower `BATCH_SIZE` / raise `GRAD_ACCUM` to keep the effective batch
  size while fitting in less VRAM
- Keep 4-bit QLoRA (already the default in `01_lora_finetune.py`'s full
  branch) — this is what makes 8B fit on 24GB in the first place
- Reduce `N_TOP_CAUSAL` / `PATCH_TOP_K_CANDIDATES` so Phase 3 patches
  fewer candidate features

None of these require touching the phase scripts — they're all in
`config.py`.

---

## Cost & time estimates (full mode, A100 80GB)

| Phase | What it does | GPU-hours | Est. cost @ $2.5/hr |
|---|---|---|---|
| 0 — Data collection | Scrape/download transcripts + prices, build counterfactuals | ~0.5h (+ ~$5 API for LLM counterfactuals, optional) | ~$1 |
| 1 — LoRA fine-tune | Domain-adapt Llama-3.1-8B on financial text | ~8h | ~$20 |
| 2 — SAE training | Train 131K-feature TopK SAE on 200M tokens | ~20-30h | ~$50-75 |
| 3 — Causal patching | Activation patching over ~5,000 candidate features | ~8-10h | ~$20-25 |
| 4 — Backtest | Extract features, build signals, backtest | ~2-4h | ~$5-10 |
| **Total** | | **~40-53h** | **~$96-136** |

Run Phase 0 and 1 first, sanity-check the merged model and perplexity
drop, *then* commit to the longer Phase 2 run — don't burn 20+ GPU-hours
before confirming Phase 1 worked.

---

## Recommended order of operations

1. Run Part A (demo mode) locally. Confirm all 4 phases complete and
   produce the expected output files.
2. Read through `circuits/results/causal_feature_library.json` and
   `backtest/results/summary.json` from the demo run just to see the
   shape of the real outputs.
3. Pick a cloud option from Part B based on budget/patience:
   - No money, willing to babysit sessions → Colab free tier or Kaggle
   - Some money, want reliability → RunPod/Lambda Labs, a few hours at a time
   - A lot of data/compute later → keep the pod running through all phases with tmux
4. Run Phase 0 and Phase 1 in full mode first (cheap, ~$20 total) and
   sanity-check results before starting the expensive Phase 2 SAE training.
5. Once Phase 2-4 finish, download `circuits/results/` and
   `backtest/results/` locally and terminate the cloud instance.
