# Changelog

## 2026-08 — reproducibility upgrade

- Reworked README around the actual research hypothesis and experimental status.
- Removed unsubstantiated novelty/certainty language from the public description.
- Added a reproducible demo dependency stack.
- Added Colab/T4 requirements.
- Added `verify_install.py` to catch Transformers/Tokenizer/TransformerLens import conflicts early.
- Added `torchao` compatibility constraint to address PEFT injection failures.
- Pinned NumPy to avoid the NumPy 2.x compatibility problems encountered in Colab.
- Improved `.gitignore` so model/data/checkpoint artifacts are not accidentally committed.
- Clarified that demo results are smoke tests, not financial evidence.
