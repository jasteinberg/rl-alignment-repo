"""
Depth profiles of the two estimators against the decoding null.

Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

Held-out AUROC per layer for the mass-mean and whitened directions, with the
per-layer random-direction null shaded from 1/2 up to its 95th percentile. A
curve inside the band is not recoverable, and height above the band is the
margin m = AUROC - p95 that the layer-selection rule maximises, so level and
clearance are readable from the same axes.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parent.parent
SWEEP = Path(os.environ.get("SNR_SWEEP", REPO / "artifacts" / "snr_sweep.json"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_depth_arms.png"

MODEL = "EleutherAI/pythia-2.8b"
PANELS = [("counterfact_true_false", "`counterfact`"), ("cities", "`cities`")]
C_PLAIN, C_WHIT, C_PERP = "#c0392b", "#2E6DA4", "#d68910"

# The rank-one corrected arm (project v_1 out of the estimator) is measured on a
# six-layer grid in the rogue-dimension sweep, not at every layer, so it is drawn
# as points rather than a curve.
ROGUE = Path(os.environ.get("ROGUE_JSON", REPO / "artifacts" / "rogue_dimension.json"))

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

S = json.load(open(SWEEP))
R = json.load(open(ROGUE))


def perp(dataset):
    node = R[dataset]
    Ls = sorted(node, key=int)
    return (np.array([float(L) for L in Ls]),
            np.array([node[L]["auroc"]["theta_perp"] for L in Ls]))


def arms(dataset):
    node = S["models"][MODEL]["datasets"][dataset]
    recs = sorted(node["layers"], key=lambda r: r["layer"])
    L = np.array([r["layer"] for r in recs], dtype=float)
    plain = np.array([r["plain"]["auroc"] for r in recs])
    whit = np.array([r["whitened"]["auroc"] for r in recs])
    null = np.array([r["null"]["auroc_p95"] for r in recs])
    return L, plain, whit, null, node["best_layer"]


def first_clear(L, arm, null):
    """First layer (excluding 0) from which the arm stays above the null."""
    idx = [i for i in range(1, len(L)) if arm[i] > null[i]]
    if not idx:
        return None
    for i in idx:
        if all(arm[j] > null[j] for j in range(i, len(L))):
            return int(L[i])
    return int(L[idx[0]])


fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)

for ax, (ds, label) in zip(axes, PANELS):
    L, plain, whit, null, best = arms(ds)
    sel = L != 0

    ax.fill_between(L[sel], 0.5, null[sel], color="#95a5a6", alpha=0.30,
                    lw=0, zorder=1, label="random-direction null")
    ax.plot(L[sel], plain[sel], color=C_PLAIN, lw=2.0, marker="o", ms=3.2,
            zorder=3, label=r"mass-mean $\hat\theta$")
    ax.plot(L[sel], whit[sel], color=C_WHIT, lw=2.0, marker="s", ms=3.2,
            zorder=3, label=r"whitened $\hat\theta_\mathrm{F}$")
    Lp, ap = perp(ds)
    ax.plot(Lp, ap, color=C_PERP, lw=0, marker="*", ms=13, mec="white",
            mew=0.6, zorder=4, label=r"rank-one corrected $\hat\theta_\perp$")
    ax.axhline(0.5, color="#bdc3c7", lw=0.8, zorder=1)

    for arm, color in [(plain, C_PLAIN), (whit, C_WHIT)]:
        Lc = first_clear(L, arm, null)
        if Lc is not None:
            ax.axvline(Lc, color=color, lw=0.9, ls="--", alpha=0.55, zorder=2)
            ax.annotate(f"clears at L{Lc}", xy=(Lc, 0.545), xytext=(3, 0),
                        textcoords="offset points", fontsize=8.5, color=color,
                        rotation=90, va="bottom")

    ax.set_xlabel("layer")
    ax.set_title(f"pythia-2.8b, {label}", fontsize=11, pad=10)
    ax.set_ylim(0.33, 1.02)

axes[0].set_ylabel("held-out AUROC")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

for ds, label in PANELS:
    L, plain, whit, null, best = arms(ds)
    print(f"{ds}: best_layer L{best}; plain clears at "
          f"{first_clear(L, plain, null)}, whitened at {first_clear(L, whit, null)}")
    for i in [26, 28, 31, 32]:
        j = int(np.where(L == i)[0][0])
        print(f"   L{i}: plain {plain[j]:.3f}  whit {whit[j]:.3f}  null {null[j]:.3f}")
