"""
Figure: the rogue dimension decides which regime a dataset is in.

Same model (pythia-2.8b), same estimator, same steering protocol -- two datasets.
On cities the plain difference-in-means direction steers correctly and whitening
degrades it, exactly as mass-mean probing intends. On counterfact_true_false at
depth the assignment inverts: plain is significantly wrong-signed and the whitened
direction is the one that steers. Reads existing artifacts/steer_ckpt/*.json only
-- NO model, NO GPU.

chi is the median across seeds (see fig_steering_signflip.py); bars are IQR.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("STEER_CKPT", REPO / "artifacts" / "steer_ckpt"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_regime_control.png"

MODEL = "pythia-2.8b"
ALPHAS = [0.5, 1.0, 2.0, 4.0]
PANELS = [
    ("cities", "cities: mass-mean steers, whitening degrades", [8, 12, 16, 20, 24, 28]),
    ("counterfact_true_false", "counterfact: the assignment inverts", [20, 24, 28]),
]

rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.55,
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
C_PLAIN = "#c0392b"
C_WHIT = "#2c6fbb"


def chi_origin(vals):
    a = np.array(ALPHAS, float)
    v = np.array(vals, float)
    return float((a * v).sum() / (a * a).sum())


def load_seeds(dataset, layer):
    return [json.load(open(f)) for f in
            sorted(glob.glob(str(CKPT / f"{MODEL}__{dataset}__L{layer}__s*.json")))]


def chi_stats(seeds, arm):
    c = np.array([chi_origin([s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS])
                  for s in seeds], float)
    med = float(np.median(c))
    q25, q75 = np.percentile(c, [25, 75])
    return med, max(med - q25, 0.0), max(q75 - med, 0.0)


fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)

for ax, (dataset, title, layers) in zip(axes, PANELS):
    have = [L for L in layers if load_seeds(dataset, L)]
    x = np.arange(len(have))
    w = 0.34
    for i, (arm, c, lab) in enumerate([("plain", C_PLAIN, "plain"),
                                       ("whitened", C_WHIT, "whitened")]):
        cent, lo, hi = [], [], []
        for L in have:
            m, a, b = chi_stats(load_seeds(dataset, L), arm)
            cent.append(m); lo.append(a); hi.append(b)
        ax.bar(x + (i - 0.5) * w, cent, w, yerr=[lo, hi], color=c, alpha=0.85,
               capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in have])
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("layer")

axes[0].set_ylabel(r"steering susceptibility $\chi=\mathrm{d}A/\mathrm{d}h|_0$")
axes[0].legend(frameon=False, fontsize=9.5)
fig.suptitle(
    "pythia-2.8b: which direction is causal depends on whether a rogue dimension dominates",
    fontsize=11.5, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

for dataset, _, layers in PANELS:
    for L in layers:
        sd = load_seeds(dataset, L)
        if not sd:
            continue
        p, _, _ = chi_stats(sd, "plain")
        w_, _, _ = chi_stats(sd, "whitened")
        npos = sum(1 for s in sd if s["alphas"]["1.0"]["plain"]["antisym"] > 0)
        print(f"{dataset:>24} L{L:<3} n={len(sd):<2} chi_plain={p:+.4f} "
              f"chi_whit={w_:+.4f}  plain +{npos}/{len(sd)}")
