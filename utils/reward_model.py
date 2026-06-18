"""
utils/reward_model.py
=====================
Bradley-Terry reward model: training, evaluation, and preference dataset utilities.

This is the toy/reference implementation used in notebooks. For full-scale
experiments use TRL's RewardTrainer against an actual LM backbone.

Usage:
    from utils.reward_model import BradleyTerry, PreferenceDataset, train_rm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np


# ── Bradley-Terry model ───────────────────────────────────────────────────────

class BradleyTerry(nn.Module):
    """
    Minimal scalar reward model for preference learning.

    Given a feature vector x (e.g. last-layer hidden state of an LM),
    outputs a scalar reward r = w^T x + b.

    The probability that trajectory sigma_1 is preferred over sigma_2:
        P[sigma_1 > sigma_2] = sigmoid(r(sigma_1) - r(sigma_2))

    Trained by minimising cross-entropy on pairwise preferences.
    """
    def __init__(self, d_input: int):
        super().__init__()
        self.head = nn.Linear(d_input, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_input) -> rewards: (batch,)"""
        return self.head(x).squeeze(-1)

    def preference_loss(
        self,
        x_w: torch.Tensor,
        x_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_w: (batch, d_input) — winning (preferred) responses
        x_l: (batch, d_input) — losing responses
        Returns scalar cross-entropy loss.
        """
        r_w = self(x_w)
        r_l = self(x_l)
        # P[w > l] = sigmoid(r_w - r_l); loss = -log P[w > l]
        return -F.logsigmoid(r_w - r_l).mean()


# ── Preference dataset ────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Holds (x_winner, x_loser) pairs of feature vectors.
    In the real pipeline these would be LM hidden states for chosen/rejected completions.
    """
    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_w, x_l = self.pairs[idx]
        return torch.tensor(x_w, dtype=torch.float32), torch.tensor(x_l, dtype=torch.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_rm(
    model: BradleyTerry,
    dataset: PreferenceDataset,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Train reward model; return loss history."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    history = {"loss": []}
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_w, x_l in loader:
            x_w, x_l = x_w.to(device), x_l.to(device)
            loss = model.preference_loss(x_w, x_l)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        history["loss"].append(epoch_loss / len(loader))

    return history


# ── Synthetic preference data ─────────────────────────────────────────────────

def make_synthetic_preferences(
    d: int = 16,
    n: int = 512,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[PreferenceDataset, np.ndarray]:
    """
    Synthetic Bradley-Terry dataset for unit-testing reward model training.

    Ground truth: r*(x) = w* . x, w* ~ N(0,1) normalised.
    Preference noise: with probability noise, the label is flipped.

    Returns (dataset, w_star) so recovery can be checked.
    """
    rng = np.random.default_rng(seed)
    w_star = rng.standard_normal(d)
    w_star /= np.linalg.norm(w_star)

    pairs = []
    for _ in range(n):
        x1 = rng.standard_normal(d)
        x2 = rng.standard_normal(d)
        r1, r2 = w_star @ x1, w_star @ x2
        # True winner is x1 if r1 > r2, with noise flip
        flip = rng.random() < noise
        if (r1 > r2) ^ flip:
            pairs.append((x1, x2))
        else:
            pairs.append((x2, x1))

    return PreferenceDataset(pairs), w_star
