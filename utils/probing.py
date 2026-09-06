"""
utils/probing.py
================
Linear-probe and signal-to-noise tooling for belief/truth directions in the
residual stream of small language models. Reproduces the mass-mean probing of Marks & Tegmark
(2023, "The Geometry of Truth") and extends it with an explicit readout-SNR
analysis in the style of Steinberg & Sompolinsky (2022).

Runs comfortably on an M-series MacBook (MPS): activation extraction on
GPT-2 / Pythia-scale models plus linear algebra on [N, d_model] arrays.
Scaffolded with assistance from Claude (Anthropic).
"""
from __future__ import annotations
import os
from typing import Optional, Sequence

import numpy as np
import torch

from utils.train import get_device

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")  # repo-local HF cache


# ---------------------------------------------------------------- model / acts
def load_lm(model_name: str = "gpt2", device: Optional[str] = None,
            cache_dir: str = DEFAULT_CACHE):
    """Load a HF causal LM + tokenizer in eval mode on the chosen device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, output_hidden_states=True,
        torch_dtype=torch.float32).to(device).eval()
    return model, tok


@torch.no_grad()
def residual_activations(model, tok, statements: Sequence[str], layer: int,
                         device: Optional[str] = None, batch_size: int = 16,
                         token: str = "last") -> np.ndarray:
    """
    Residual-stream activations at `layer` -> [N, d_model].
    hidden_states index: 0 = embeddings, L = output of block L (12-layer GPT-2
    -> layers 0..12). The last entry has already passed through the model's
    final norm, so it is not comparable to the others in scale -- verified by
    lm_head(hidden_states[-1]) reproducing the returned logits.
    token="last" takes the last non-pad position.
    """
    device = device or get_device()
    out = []
    for i in range(0, len(statements), batch_size):
        enc = tok(list(statements[i:i + batch_size]), return_tensors="pt",
                  padding=True, truncation=True).to(device)
        hs = model(**enc).hidden_states[layer]                 # [B, T, d]
        if token == "last":
            idx = (enc["attention_mask"].sum(1) - 1).clamp(min=0)
            acts = hs[torch.arange(hs.size(0)), idx]
        elif token == "mean":
            m = enc["attention_mask"].unsqueeze(-1)
            acts = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            raise ValueError(f"unknown token mode {token!r}")
        out.append(acts.float().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------- probes / SNR
def mass_mean_direction(X: np.ndarray, y: np.ndarray, whiten: bool = False,
                        shrink: bool = False):
    """
    theta = mean(X[y==1]) - mean(X[y==0]).  With whiten=True, premultiply by the
    inverse within-class covariance (the Fisher/LDA direction).  With shrink=True
    the covariance is estimated by Ledoit-Wolf shrinkage -- well-conditioned when
    n << d, where the raw empirical covariance is singular -- otherwise the
    empirical covariance with a small ridge is used.
    Returns (unit theta, midpoint bias on theta^T x).
    """
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    theta = mu1 - mu0
    if whiten:
        Xc = np.vstack([X[y == 1] - mu1, X[y == 0] - mu0])
        if shrink:
            from sklearn.covariance import LedoitWolf
            theta = LedoitWolf().fit(Xc).precision_ @ theta
        else:
            cov = np.cov(Xc, rowvar=False) + 1e-3 * np.eye(X.shape[1])
            theta = np.linalg.solve(cov, theta)
    theta = theta / (np.linalg.norm(theta) + 1e-12)
    return theta, float(0.5 * (mu1 + mu0) @ theta)


def direction_snr(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> dict:
    """
    Readout SNR along theta (the truth-direction analogue of S&S 2022):
    d'^2 = (m1-m0)^2 / (0.5(s1^2+s0^2)). Also returns Fisher ratio and separation.
    """
    z = X @ theta
    z1, z0 = z[y == 1], z[y == 0]
    m1, m0, s1, s0 = z1.mean(), z0.mean(), z1.var(), z0.var()
    dprime2 = (m1 - m0) ** 2 / (0.5 * (s1 + s0) + 1e-12)
    return {"snr_dprime2": float(dprime2), "dprime": float(np.sqrt(max(dprime2, 0.0))),
            "fisher": float((m1 - m0) ** 2 / (s1 + s0 + 1e-12)), "sep": float(m1 - m0)}


def probe_accuracy(X, y, theta, bias) -> float:
    return float(((X @ theta > bias).astype(int) == y).mean())


def project_out(X: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Remove the subspace spanned by dirs ([k, d]) from X ([N, d])."""
    Q, _ = np.linalg.qr(dirs.T)
    return X - (X @ Q) @ Q.T


def layer_sweep(model, tok, statements, labels, device=None, whiten=False,
                shrink=False, token="last", split=0.5, seed=0) -> dict:
    """Per-layer held-out SNR + probe accuracy from a mass-mean probe."""
    device = device or get_device()
    y = np.asarray(labels).astype(int)
    perm = np.random.default_rng(seed).permutation(len(y))
    tr, te = perm[:int(split * len(y))], perm[int(split * len(y)):]
    snr, acc = [], []
    for L in range(model.config.num_hidden_layers + 1):
        X = residual_activations(model, tok, statements, L, device, token=token)
        theta, bias = mass_mean_direction(X[tr], y[tr], whiten=whiten, shrink=shrink)
        snr.append(direction_snr(X[te], y[te], theta)["snr_dprime2"])
        acc.append(probe_accuracy(X[te], y[te], theta, bias))
    return {"layer": np.arange(model.config.num_hidden_layers + 1),
            "snr_dprime2": np.array(snr), "accuracy": np.array(acc)}
