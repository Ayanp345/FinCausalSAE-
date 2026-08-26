"""Small environment check for FinCausalSAE."""
import importlib

packages = [
    "torch",
    "transformers",
    "tokenizers",
    "peft",
    "transformer_lens",
    "numpy",
    "pandas",
]

for name in packages:
    mod = importlib.import_module(name)
    print(f"{name:18s} {getattr(mod, '__version__', 'unknown')}")

import torch
print("torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
print("Transformer/Tokenizer import: OK")
print("TransformerLens import: OK")
print("FinCausalSAE environment: READY")
