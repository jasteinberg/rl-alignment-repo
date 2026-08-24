"""
Figure: depth profiles of the plausibility axis and the truth axis.

Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

Ordinate is the MARGIN m = held-out AUROC - the layer's own random-direction
null p95, the same quantity the layer-selection rule maximises, because the null
widens with depth and raw AUROC is not comparable across layers. Abscissa is
fractional depth L / L_max so models of different depth are commensurable.

Left: pythia-2.8b, where `likely` peaks early (L12 of 32) and `cities` late
(L29), i.e. at opposite ends of the depth axis. Right: the same two curves at
410m and 1.4b, where the ordering does NOT hold -- `cities` peaks at L11 and L7
of 24, BEFORE `likely` at L15 and L13, and its late-depth structure appears only
as a secondary rise over the final layers.

Layer 0 is the embedding, excluded from selection, drawn hollow. On templated
statements its last-token activation is nearly constant within a dataset, so
`cities` is degenerate there; `likely` is not, and that is the point of the
early-depth end of the left panel.

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
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
SWEEP = Path(os.environ.get("SNR_SWEEP", REPO / "artifacts" / "snr_sweep.json"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_plausibility_depth.png"

C_LIKELY, C_TRUTH = "#8e44ad", "#c0392b"
BIG = "EleutherAI/pythia-2.8b"
SMALL = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b"]
STYLE = {"EleutherAI/pythia-410m": ":", "EleutherAI/pythia-1.4b": "-"}

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


def profile(model, dataset):
    node = S["models"][model]["datasets"][dataset]
    recs = sorted(node["layers"], key=lambda r: r["layer"])
    L = np.array([r["layer"] for r in recs], dtype=float)
    m = np.array([r["plain"]["auroc"] - r["null"]["auroc_p95"] for r in recs])
    return L / L.max(), m, L, node["best_layer"]


fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)

# ---- left: pythia-2.8b -------------------------------------------------
ax = axes[0]
for ds, label, color in [("likely", "likely (plausibility)", C_LIKELY),
                         ("cities", "cities (truth)", C_TRUTH)]:
    x, m, L, best = profile(BIG, ds)
    sel = L != 0
    ax.plot(x[sel], m[sel], lw=2.0, color=color, marker="o", ms=3.4,
            label=label, zorder=3)
    ax.plot(x[~sel], m[~sel], lw=0, marker="o", ms=5, mfc="white",
            mec=color, zorder=3)
    j = int(np.where(L == best)[0][0])
    ax.plot([x[j]], [m[j]], marker="*", ms=15, lw=0, color="#d68910", zorder=5)
    ax.annotate(f"L{best}", xy=(x[j], m[j]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9,
                color=color)

ax.axhline(0.0, color="#bdc3c7", lw=0.8, zorder=1)
ax.set_ylabel(r"margin $m = \mathrm{AUROC} - p_{95}$")
ax.set_xlabel("fractional depth $L/L_{\\max}$")
ax.set_title("pythia-2.8b: peaks at opposite ends", fontsize=11, pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.annotate("L0 = embedding\n(excluded)", xy=(0.0, 0.113), xytext=(0.055, 0.035),
            fontsize=8.5, color="#7f8c8d",
            arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=0.8))

# ---- right: 410m and 1.4b ---------------------------------------------
ax = axes[1]
for model in SMALL:
    tag = model.split("-")[-1]
    for ds, color in [("likely", C_LIKELY), ("cities", C_TRUTH)]:
        x, m, L, best = profile(model, ds)
        sel = L != 0
        ax.plot(x[sel], m[sel], lw=1.8, ls=STYLE[model], color=color,
                zorder=3)
        ax.plot(x[~sel], m[~sel], lw=0, marker="o", ms=5, mfc="white",
                mec=color, zorder=3)
        j = int(np.where(L == best)[0][0])
        ax.plot([x[j]], [m[j]], marker="*", ms=13, lw=0, color="#d68910",
                zorder=5)
        ax.annotate(f"L{best}", xy=(x[j], m[j]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=color)

ax.axhline(0.0, color="#bdc3c7", lw=0.8, zorder=1)
ax.set_xlabel("fractional depth $L/L_{\\max}$")
ax.set_title("410m and 1.4b: the ordering reverses", fontsize=11, pad=10)
handles = [Line2D([], [], color="#7f8c8d", lw=1.8, ls=STYLE[m_], label=t)
           for m_, t in zip(SMALL, ["410m", "1.4b"])]
ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right",
          title="model", title_fontsize=9.5)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}\n")

for model in [BIG] + SMALL:
    print(model)
    for ds in ["likely", "cities"]:
        x, m, L, best = profile(model, ds)
        j = int(np.where(L == best)[0][0])
        k = int(np.argmax(np.where(L == 0, -np.inf, m)))
        print(f"  {ds:<7} best_layer L{best:>2} (m={m[j]:+.3f}, "
              f"depth {x[j]:.2f})   argmax m = L{int(L[k])} ({m[k]:+.3f})")
        print("     last three layers: " + ", ".join(
            f"L{int(l)} {v:+.3f}" for l, v in zip(L[-3:], m[-3:])))
