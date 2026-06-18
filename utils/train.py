"""
utils/train.py
==============
Generic training loop and plotting utilities for RL-alignment experiments.
Mirrors the style of interp-foundations/utils/train.py.

Usage:
    from utils.train import plot_history, smooth
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():    return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── Plotting ──────────────────────────────────────────────────────────────────

def smooth(xs: List[float], window: int = 50) -> np.ndarray:
    """Simple moving average for noisy RL curves."""
    xs = np.array(xs)
    if len(xs) < window:
        return xs
    return np.convolve(xs, np.ones(window) / window, mode="valid")


def plot_history(
    history: Dict[str, List[float]],
    keys: Optional[List[str]] = None,
    title: str = "",
    smooth_window: int = 50,
    figsize=(10, 3),
) -> None:
    """
    Plot training curves from a history dict.
    Keys default to all keys in history.
    """
    keys = keys or list(history.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        raw = history[k]
        ax.plot(raw, alpha=0.3, color="steelblue", label="raw")
        if len(raw) >= smooth_window:
            ax.plot(smooth(raw, smooth_window), color="steelblue", label=f"smooth ({smooth_window})")
        ax.set_title(k)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_histories(
    histories: Dict[str, Dict[str, List[float]]],
    keys: Union[str, List[str]] = "returns",
    smooth_window: int = 20,
    normalize_x: bool = True,
    title: str = "",
    figsize: Optional[tuple] = None,
) -> None:
    """
    Overlay one or more metrics from multiple algorithm histories.

    One subplot per key; within each subplot, one curve per algorithm that
    actually has that key (algorithms missing the key are silently skipped —
    e.g. REINFORCE has no `ratio_mean`, only PPO does).

    Useful when algorithms differ in x-axis units (e.g. REINFORCE has
    episodes, PPO has iterations) — `normalize_x=True` rescales each curve
    to training-progress fraction in [0, 1] so they're visually comparable.

    Parameters
    ----------
    histories : dict[str, dict[str, list[float]]]
        Mapping from algorithm label (used in the legend) to its history dict.
    keys : str | list[str]
        Which metric(s) to plot (default "returns"). String for one subplot;
        list for one subplot per key.
    smooth_window : int
        Moving-average window.
    normalize_x : bool
        If True (default), plot over x ∈ [0, 1]. If False, use raw index —
        only meaningful when x-axis units actually match across algorithms.
    title : str
        Optional figure-level title.
    figsize : tuple | None
        Figure size; defaults to (4 * n_keys, 3.5).
    """
    if isinstance(keys, str):
        keys = [keys]
    if figsize is None:
        figsize = (4 * len(keys), 3.5)

    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        for label, history in histories.items():
            if key not in history:
                continue
            ys = smooth(history[key], window=smooth_window)
            xs = np.linspace(0, 1, len(ys)) if normalize_x else np.arange(len(ys))
            ax.plot(xs, ys, label=label)
        ax.set_xlabel("training progress" if normalize_x else "step")
        ax.set_ylabel(f"{key} (smoothed)")
        ax.set_title(key)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
