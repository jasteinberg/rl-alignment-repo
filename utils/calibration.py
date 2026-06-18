"""
utils/calibration.py
====================
Calibration metrics and reliability diagrams.

Usage:
    from utils.calibration import ece, reliability_diagram, temperature_scale
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Partitions predictions into n_bins by confidence, then measures
    the weighted average gap between mean confidence and mean accuracy:

        ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Args:
        probs:  (n,) predicted probability of the positive class
        labels: (n,) binary ground truth
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece_val  = 0.0
    n        = len(probs)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return float(ece_val)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plot reliability diagram (calibration curve).
    Perfect calibration = diagonal line.
    """
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_mids   = (bins[:-1] + bins[1:]) / 2
    accs, confs, counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
        else:
            accs.append(labels[mask].mean())
            confs.append(probs[mask].mean())
            counts.append(mask.sum())

    accs   = np.array(accs)
    confs  = np.array(confs)
    counts = np.array(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Reliability diagram
    ax1.plot([0,1], [0,1], "k--", lw=1, label="perfect calibration")
    valid = ~np.isnan(accs)
    ax1.bar(bin_mids[valid], accs[valid], width=0.08, alpha=0.6,
            color="steelblue", label="accuracy")
    ax1.bar(bin_mids[valid], confs[valid], width=0.08, alpha=0.3,
            color="red", label="confidence")
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.legend(fontsize=8)
    ax1.set_title(f"{title}\nECE = {ece(probs, labels, n_bins):.4f}")

    # Confidence histogram
    ax2.bar(bin_mids, counts, width=0.08, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")

    plt.tight_layout(); plt.show()


def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    T_range: Tuple[float, float] = (0.1, 5.0),
    n_grid: int = 100,
) -> float:
    """
    Find optimal temperature T* that minimises NLL on (logits, labels).
    Temperature scaling: p = sigmoid(logit / T).

    Returns T*.
    """
    Ts    = np.linspace(*T_range, n_grid)
    nlls  = []
    for T in Ts:
        p    = torch.sigmoid(torch.tensor(logits / T, dtype=torch.float32))
        y    = torch.tensor(labels,  dtype=torch.float32)
        nll  = F.binary_cross_entropy(p, y).item()
        nlls.append(nll)
    return float(Ts[np.argmin(nlls)])


# Import F for temperature_scale
import torch.nn.functional as F
