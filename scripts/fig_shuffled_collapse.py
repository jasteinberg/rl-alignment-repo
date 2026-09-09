"""
Two panels for *How small is too small?*

Left: the pooled shuffled-label law. In-sample excess AUROC of the mass-mean
direction under shuffled labels on counterfact, four Pythia widths, against
N/2d, with the pooled fit 0.045 (N/2d)^-0.49. Points with non-positive excess
(three at pythia-70m) are dropped from the log fit and not drawn.

Right: the effective-dimension collapse. The same excess on pythia-2.8b for
four datasets at their own best layer, against N/PR, with the parameter-free
prediction (1/sqrt(pi)) sqrt(PR/N). Three datasets fall on the line across a
thirtyfold range in N; sp_en_trans sits ~40% above it.

Usage: FIGURE_DIR=... python scripts/fig_shuffled_collapse.py

Drafted with the assistance of Claude (Anthropic).
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parent.parent
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))
COVER = json.load(open(REPO / "artifacts" / "cover_by_dataset.json"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_shuffled_collapse.png"

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

MODEL_STYLE = {
    "EleutherAI/pythia-70m":  ("70m",  "o", "#95a5a6"),
    "EleutherAI/pythia-410m": ("410m", "s", "#7f8c8d"),
    "EleutherAI/pythia-1.4b": ("1.4b", "^", "#34495e"),
    "EleutherAI/pythia-2.8b": ("2.8b", "D", "#c0392b"),
}
DS_STYLE = {
    "counterfact_true_false": ("counterfact", "D", "#c0392b"),
    "cities":                 ("cities",      "o", "#2E6DA4"),
    "larger_than":            ("larger_than", "s", "#d68910"),
    "sp_en_trans":            ("sp_en_trans", "^", "#7d3c98"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- left: pooled law across models -------------------------------------
xs, ys = [], []
for m, (lab, mk, col) in MODEL_STYLE.items():
    rows = SWEEP["models"][m]["cover"]
    x = np.array([r["N_over_2d"] for r in rows])
    y = np.array([r["shuffled"]["auroc"] - 0.5 for r in rows])
    keep = y > 0
    ax1.scatter(x[keep], y[keep], marker=mk, s=34, color=col, label=f"pythia-{lab}", zorder=3)
    xs.append(x[keep]); ys.append(y[keep])
x = np.concatenate(xs); y = np.concatenate(ys)
b, la = np.polyfit(np.log(x), np.log(y), 1)
xx = np.geomspace(x.min() * 0.7, x.max() * 1.4, 100)
ax1.plot(xx, math.exp(la) * xx ** b, color="k", lw=1.1,
         label=rf"pooled fit $0.045\,(N/2d)^{{-0.49}}$")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"$N/2d$")
ax1.set_ylabel(r"in-sample $\mathrm{AUROC}_{\mathrm{shuffled}} - 1/2$")
ax1.set_title("Shuffled labels still separate, and it scales as $N^{-1/2}$", fontsize=11)
ax1.axvline(1.0, color="k", lw=0.7, ls="--", alpha=0.6)
ax1.text(1.05, 0.3, "Cover's capacity\n$N = 2d$", fontsize=8.5, va="top")
ax1.legend(fontsize=8.5, loc="lower left", frameon=False)

# ---- right: the collapse ------------------------------------------------
C0 = 1.0 / math.sqrt(math.pi)
for ds, (lab, mk, col) in DS_STYLE.items():
    v = COVER["results"][ds]
    PR = v["spectrum"]["participation_ratio"]
    N = np.array([r["N"] for r in v["curve"]])
    exc = np.array([r["excess_mean"] for r in v["curve"]])
    sd = np.array([r["excess_sd"] for r in v["curve"]]) / np.sqrt(v["curve"][0]["n_rep"])
    ax2.errorbar(N / PR, exc, yerr=sd, fmt=mk, ms=6, color=col, capsize=2, lw=0.8,
                 label=rf"{lab}  (PR = {PR:.1f})", zorder=3)
xx = np.geomspace(2.0, 400.0, 100)
ax2.plot(xx, C0 * xx ** -0.5, color="k", lw=1.1,
         label=r"$\pi^{-1/2}\,\sqrt{\mathrm{PR}/N}$, no free parameter")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel(r"$N/\mathrm{PR}$")
ax2.set_ylabel(r"in-sample $\mathrm{AUROC}_{\mathrm{shuffled}} - 1/2$")
ax2.set_title("Effective dimension collapses the amplitude (pythia-2.8b)", fontsize=11)
ax2.legend(fontsize=8.5, loc="lower left", frameon=False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"pooled fit: a={math.exp(la):.4f} b={b:.3f}")
