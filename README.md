# FinCausalSAE

### Causal Sparse Autoencoder Features for Financial Language Models

FinCausalSAE is an experimental mechanistic-interpretability pipeline for studying whether internal features of a language model are merely **correlated** with financial signals or whether intervening on those features can **causally change** a model's financial judgment.

The project combines four stages:

1. **Financial text + counterfactuals** — build earnings-call examples and controlled text pairs.
2. **Domain adaptation** — optionally LoRA-fine-tune a base language model on financial text.
3. **Sparse autoencoder analysis** — learn sparse features from an internal transformer layer.
4. **Causal interventions + backtesting** — patch candidate features, measure changes in model outputs, and compare the resulting signal with correlational and FinBERT-style baselines.

> **Important:** this repository is a research prototype, not evidence that any discovered feature is genuinely causal until the intervention results have been validated on held-out data and against appropriate controls.

---

## Research question

The central question is:

> **Can sparse, interpretable features identified inside a financial language model be distinguished by causal interventions from features that only correlate with financial outcomes?**

The key experimental idea is to construct paired examples such as:

- margin beat → margin miss
- guidance raised → guidance lowered
- uncertainty/hedging → more certain wording
- negative tone → neutralized wording

The model's internal activations are then compared across the pair. Candidate SAE features are selected and individually intervened on. A useful feature should produce a measurable change in the model's bullish/bearish decision when its activation is patched, while appropriate controls should not show the same effect.

---

## Pipeline

```text
                    FinCausalSAE
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   Phase 0: Data                    Counterfactuals
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             Phase 1: Domain Adaptation
                  GPT-2 demo / Llama full
                         │
                         ▼
              Phase 2: Sparse Features
              SimpleTopK / SAELens
                         │
                         ▼
              Phase 3: Causal Patching
              feature intervention → Δlogit
                         │
                         ▼
               Phase 4: Backtesting
        causal vs correlational vs FinBERT
```

### Demo mode

The demo uses GPT-2 and a small synthetic corpus so the pipeline can be tested on a laptop or free cloud notebook. It is a **software and experimental smoke test**, not a meaningful financial result.

### Full mode

The intended research configuration uses a Llama-3.1-8B base model, real financial text/market data, a larger SAE, and GPU compute. The full run requires substantially more compute and should be treated as a separate research experiment.

---

## What is implemented

| Component | Demo | Full |
|---|---:|---:|
| Synthetic financial corpus | ✓ | — |
| Counterfactual generation | ✓ | ✓ |
| LoRA fine-tuning | ✓ | ✓ |
| Sparse autoencoder | Lightweight TopK SAE | SAELens TopK SAE |
| Activation patching | ✓ | ✓ |
| Causal feature ranking | ✓ | ✓ |
| Portfolio backtest | ✓ | ✓ |
| FinBERT baseline | Optional | Optional |

---

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install the tested demo stack

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

### 3. Verify the environment

```bash
python verify_install.py
```

### 4. Run the demo

```bash
python data/00_collect_data.py --mode demo
python models/01_lora_finetune.py --mode demo
python sae/02_train_sae.py --mode demo
python circuits/03_causal_patching.py --mode demo
python backtest/04_portfolio_backtest.py --mode demo
```

Or use:

```bash
bash run_demo.sh
```

On Windows, run the individual Python commands above from PowerShell.

---

## Google Colab / free GPU workflow

A free T4-class GPU is enough for the **demo** and for development/debugging. The demo configuration is deliberately small: GPT-2, 6,144 SAE features, and 200k training tokens.

In Colab:

```python
!nvidia-smi
!python -m pip install -r requirements-demo.txt
!python verify_install.py
```

Then run the five phases in order.

For the full experiment, use a dedicated GPU runtime and expect much higher memory, storage, and runtime requirements.

---

## Repository structure

```text
fincausalsae/
├── README.md
├── SETUP_GUIDE.md
├── requirements-demo.txt
├── requirements.txt
├── requirements-colab.txt
├── verify_install.py
├── run_demo.sh
├── config.py
├── data/
│   └── 00_collect_data.py
├── models/
│   └── 01_lora_finetune.py
├── sae/
│   └── 02_train_sae.py
├── circuits/
│   └── 03_causal_patching.py
├── backtest/
│   └── 04_portfolio_backtest.py
└── utils/
    ├── cli.py
    └── simple_sae.py
```

Runtime outputs are written to `data/processed/`, `data/counterfactuals/`, `models/fin-lora/`, `sae/checkpoints/`, `circuits/results/`, and `backtest/results/`.

These generated artifacts should **not** be committed to GitHub unless they are intentionally selected as research artifacts.

---

## Compatibility and reproducibility

The original prototype used unconstrained dependency ranges. That can cause a Colab environment to mix incompatible versions of PyTorch, Transformers, Tokenizers, PEFT, NumPy, and TransformerLens.

This version therefore includes a **tested compatibility stack** for the demo environment. In particular, keep the core packages together rather than upgrading one package independently.

If an old notebook session has already installed conflicting packages, the safest fix is to restart the runtime and install the pinned requirements in a fresh session.

---

## Full research configuration

The full configuration is defined centrally in `config.py`:

- base model: `meta-llama/Llama-3.1-8B`
- target layer: 20
- SAE width: 131,072 features
- training tokens: 200M
- longer context and larger batches
- real financial data

The exact compute requirement depends on implementation details, sequence lengths, checkpointing, and GPU type. The numbers in this repository should therefore be treated as planning estimates rather than guarantees.

---

## Evaluation philosophy

A strong result should survive more than one test. Recommended checks include:

1. **Held-out examples** — discover features on training data and evaluate on unseen calls.
2. **Counterfactual consistency** — the same feature should respond predictably across multiple independently generated pairs.
3. **Intervention controls** — compare real patches against shuffled, random-feature, and sign-reversed controls.
4. **Ablation/recovery tests** — remove a candidate feature and test whether the predicted behavioral change occurs.
5. **Alternative baselines** — compare against raw activation probes, correlational SAE features, and FinBERT/lexicon baselines.
6. **Temporal validation** — avoid using future market information when constructing features or selecting hyperparameters.
7. **Multiple seeds** — report variability rather than a single favorable run.

The backtest should be interpreted as an evaluation of the experimental signal, not as investment advice.

---

## Limitations

- The demo corpus is synthetic and intentionally easy; it cannot establish a real-world financial effect.
- Financial language is highly contextual, so simple counterfactual rewrites may introduce unintended changes.
- Activation patching can establish evidence for causal influence under a particular intervention, but it does not by itself establish a human-interpretable mechanism.
- Market returns contain many confounders unrelated to earnings-call language.
- The project currently uses a relatively small set of controlled counterfactual transformations.
- Full-mode claims require careful held-out evaluation and statistical testing.

---

## Roadmap

- [x] Counterfactual corpus generation
- [x] Lightweight TopK SAE for local development
- [x] Causal feature patching prototype
- [x] Correlational and FinBERT-style baselines
- [ ] Larger real-world financial corpus
- [ ] Stronger counterfactual generation and validation
- [ ] Random-feature and shuffled-patch controls
- [ ] Multi-seed causal-effect confidence intervals
- [ ] Out-of-time financial evaluation
- [ ] Reproducible full-scale Llama + SAELens experiment

---

## Citation

If this repository develops into a paper, replace the placeholder below with the final bibliographic entry:

```bibtex
@software{fincausalsae,
  title  = {FinCausalSAE: Causal Sparse Autoencoder Features for Financial Language Models},
  author = {Ayan Panja},
  year   = {2026},
  note   = {Research prototype}
}
```

## License

No license has been declared yet. Until a license is added, treat the repository as **all rights reserved** and do not assume that code may be redistributed or reused commercially.
