"""
Figure: layer sweep -- held-out AUROC of the mass-mean direction and its
random-direction null as a function of depth, pythia-2.8b / cities.

Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

The point is the MARGIN, not the curve: the null itself widens with depth
(activations grow more anisotropic), so a deep-layer AUROC in the low 0.7s can
sit inside a null that an early layer would have cleared. Selection maximises
AUROC above the layer's OWN null -- the artifact's `best_layer`, marked by the
star. That is NOT the same as the peak of the curve, and quoting the peak is the
error the Jul-8 figure encoded.

Layer 0 is the embedding; on templated statements the last-token embedding is
nearly constant within a dataset, so its d' is degenerate. Excluded from
selection, drawn hollow.

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
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_layer_sweep.png"

MODEL = os.environ.get("SWEEP_MODEL", "EleutherAI/pythia-2.8b")
DATASET = os.environ.get("SWEEP_DS", "cities")

C_PLAIN, C_NULL = "#c0392b", "#7f8c8d"

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
node = S["models"][MODEL]["datasets"][DATASET]
recs = sorted(node["layers"], key=lambda r: r["layer"])
best_layer = node["best_layer"]

L = np.array([r["layer"] for r in recs])
auroc = np.array([r["plain"]["auroc"] for r in recs])
null95 = np.array([r["null"]["auroc_p95"] for r in recs])
sel = L != 0

fig, ax = plt.subplots(figsize=(6.8, 4.2))

ax.fill_between(L[sel], 0.5, null95[sel], color=C_NULL, alpha=0.18, zorder=1,
                label="random-direction null (to $p_{95}$)")
ax.plot(L[sel], null95[sel], color=C_NULL, lw=1.0, ls="--", zorder=2)
ax.plot(L[sel], auroc[sel], marker="o", ms=4, lw=2.0, color=C_PLAIN, zorder=3,
        label=r"mass-mean $\hat\theta$")

if (~sel).any():
    ax.plot(L[~sel], auroc[~sel], marker="o", ms=5, lw=0, mfc="white",
            mec=C_PLAIN, zorder=3)
    ax.annotate("L0 = embedding\n(excluded)", xy=(0, auroc[~sel][0]),
                xytext=(1.2, 0.56), fontsize=8.5, color="#7f8c8d",
                arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=0.8))

b = next(r for r in recs if r["layer"] == best_layer)
ax.plot([best_layer], [b["plain"]["auroc"]], marker="*", ms=15, lw=0,
        color="#d68910", zorder=5, label=f"best margin: L{best_layer}")

ax.axhline(0.5, color="#bdc3c7", lw=0.8, zorder=1)
ax.set_xlabel("layer")
ax.set_ylabel("held-out AUROC")
ax.set_ylim(0.45, 1.02)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("Recoverability is the margin above the null "
             f"({MODEL.split('/')[-1]}, {DATASET})", fontsize=11, pad=10)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}\n")

m_best = b["plain"]["auroc"] - b["null"]["auroc_p95"]
print(f"best_layer (artifact) = L{best_layer}  AUROC={b['plain']['auroc']:.3f}  "
      f"null95={b['null']['auroc_p95']:.3f}  margin={m_best:+.3f}")
peak = max((r for r in recs if r["layer"] != 0), key=lambda r: r["plain"]["auroc"])
print(f"peak AUROC (unpenalised) = L{peak['layer']}: {peak['plain']['auroc']:.3f}"
      "   <- NOT the selection rule")
print(f"null p95 over depth: {null95[sel].min():.3f} -> {null95[sel].max():.3f}")
print("\nlayer  auroc  null95  margin")
for r in recs:
    m = r["plain"]["auroc"] - r["null"]["auroc_p95"]
    tag = "  <- best margin" if r["layer"] == best_layer else (
        "  (embedding, excluded)" if r["layer"] == 0 else "")
    print(f"{r['layer']:>5} {r['plain']['auroc']:>6.3f} "
          f"{r['null']['auroc_p95']:>7.3f} {m:>+7.3f}{tag}")
