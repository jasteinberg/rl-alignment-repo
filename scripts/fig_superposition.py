"""
Figure: the superposition probe -- d' of the mass-mean direction after projecting
out the top-k principal components of the activations. Reads the existing
artifacts/snr_sweep.json only -- NO model, NO GPU.

Reads as a salience knob. If a truth direction merely rides the high-variance
subspace, removing the leading components destroys it; if it occupies its own
low-variance subspace, d' survives. counterfact_true_false does neither: d' RISES
as the top directions are stripped, because the leading eigendirection is the
massive-activation axis the mass-mean estimator has collapsed onto -- the
rogue-dimension diagnosis, visible here before it is named.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parent.parent
SWEEP = Path(os.environ.get("SNR_SWEEP", REPO / "artifacts" / "snr_sweep.json"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_superposition.png"

MODEL = "EleutherAI/pythia-2.8b"
SERIES = [
    ("cities", "cities", "#7f8c8d", False),
    ("neg_cities", "neg_cities", "#95a5a6", False),
    ("larger_than", "larger_than", "#aab3b5", False),
    ("counterfact_true_false", "counterfact", "#c0392b", True),
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

S = json.load(open(SWEEP))
node = S["models"][MODEL]["datasets"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ks_ref = None
for key, label, colour, emph in SERIES:
    sp = node[key]["superposition"]
    ks = [r["k"] for r in sp]
    dp = [r["d_prime"] for r in sp]
    if ks_ref is None:
        ks_ref = ks
    x = np.arange(len(ks))
    ax.plot(x, dp,
            marker="o",
            markersize=5.5 if emph else 4,
            linewidth=2.2 if emph else 1.2,
            color=colour,
            zorder=3 if emph else 2,
            label=label)

x = np.arange(len(ks_ref))
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in ks_ref])
ax.set_xlabel(r"top-$k$ principal components removed")
ax.set_ylabel(r"separation $d'$ of $\hat\theta$")
ax.axhline(0, color="#bdc3c7", linewidth=0.8, zorder=1)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title("The salience knob (pythia-2.8b, best layer per dataset)",
             fontsize=11, pad=10)

sp = node["counterfact_true_false"]["superposition"]
d0, dlast = sp[0]["d_prime"], sp[-1]["d_prime"]
ax.annotate(f"rises: {d0:.2f}" + r"$\rightarrow$" + f"{dlast:.2f}",
            xy=(len(ks_ref) - 1, dlast),
            xytext=(len(ks_ref) - 3.1, dlast + 0.62),
            fontsize=9.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

for key, label, _, _ in SERIES:
    sp = node[key]["superposition"]
    row = "  ".join(f"k={r['k']}:{r['d_prime']:.2f}" for r in sp)
    print(f"{label:>14}  {row}")
