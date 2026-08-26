"""
Minimal TopK sparse autoencoder used in DEMO mode, shared by phases 2/3/4
so all of them reconstruct the exact same architecture from a saved
state_dict + metadata JSON.

API surface deliberately mirrors the subset of SAELens's `SAE` class that
this project actually uses (`encode`, `decode`, `.W_dec`), so Phase 3/4
code does not need to branch on which SAE implementation is active.
"""

import json
import torch
import torch.nn as nn


class SimpleTopKSAE(nn.Module):
    def __init__(self, d_in, n_features, k):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) * (1.0 / d_in ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_in) * (1.0 / n_features ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        pre = torch.relu(pre)
        if self.k < self.n_features:
            topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
            sparse = torch.zeros_like(pre)
            sparse.scatter_(-1, topk_idx, topk_vals)
            return sparse
        return pre

    def decode(self, feats):
        return feats @ self.W_dec + self.b_dec

    def forward(self, x):
        feats = self.encode(x)
        return self.decode(feats), feats


def load_demo_sae(sae_dir):
    """Reconstructs a SimpleTopKSAE from the state_dict + meta JSON saved by Phase 2."""
    sae_dir = str(sae_dir)
    with open(f"{sae_dir}/sae_meta.json") as f:
        meta = json.load(f)
    sae = SimpleTopKSAE(meta["d_in"], meta["n_features"], meta["k"])
    state = torch.load(f"{sae_dir}/fin_sae_state.pt", map_location="cpu")
    sae.load_state_dict(state)
    sae.eval()
    return sae
