"""
Figure: cross-dataset transfer of the mass-mean direction, pythia-2.8b.
Held-out AUROC for a direction fit on each dataset (rows), evaluated on each
(columns).

Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

The off-diagonal is ORGANISED, not merely weak, which is why the colour map is
diverging and centred on chance rather than sequential: the below-chance cells
ARE the finding. Between a statement type and its logical negation the same
vector reads truth BACKWARDS (cells near 0.07) -- a different thing from failing
to transfer (cells near 0.5).

TRANSFER_FULL=1 draws all nine main-tier datasets instead of the four-dataset
polarity block. Worth running before claiming anything about "unrelated
families": in the 9x9, cities -> common_claim is ~0.71, which is not chance.

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
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_transfer.png"

MODEL = os.environ.get("TRANSFER_MODEL", "EleutherAI/pythia-2.8b")
FULL = os.environ.get("TRANSFER_FULL") == "1"

POLARITY = ["cities", "neg_cities", "larger_than", "smaller_than"]
SHORT = {
    "cities": "cities",
    "neg_cities": "neg_cities",
    "larger_than": "larger_than",
    "smaller_than": "smaller_than",
    "cities_cities_conj": "cc_conj",
    "cities_cities_disj": "cc_disj",
    "common_claim_true_false": "common_claim",
    "companies_true_false": "companies",
    "counterfact_true_false": "counterfact",
}

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

S = json.load(open(SWEEP))
T = S["models"][MODEL]["transfer"]

keys = [k for k in SHORT if k in T] if FULL else POLARITY
M = np.array([[T[r][c] for c in keys] for r in keys])

fig, ax = plt.subplots(figsize=(7.4, 6.0) if FULL else (5.9, 5.0))
im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0)

ax.set_xticks(np.arange(len(keys)))
ax.set_yticks(np.arange(len(keys)))
ax.set_xticklabels([SHORT[k] for k in keys], rotation=45, ha="right", fontsize=9)
ax.set_yticklabels([SHORT[k] for k in keys], fontsize=9)
ax.set_xlabel("evaluated on")
ax.set_ylabel("fit on")

for i in range(len(keys)):
    for j in range(len(keys)):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v - 0.5) > 0.30 else "#2c3e50",
                fontweight="bold" if (i == j or v < 0.2) else "normal")

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("held-out AUROC", fontsize=9.5)
cb.ax.axhline(0.5, color="#2c3e50", lw=1.0)

ax.set_title(f"Transfer is polarity- and family-structured ({MODEL.split('/')[-1]})",
             fontsize=11, pad=10)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}\n")

diag = [T[k][k] for k in keys]
print(f"diagonal: {min(diag):.3f} - {max(diag):.3f}")
print("\nnegation pairs (the anti-transfer finding):")
for a, b in [("cities", "neg_cities"), ("larger_than", "smaller_than")]:
    if a in T and b in T:
        print(f"  {a:>12} -> {b:<14} {T[a][b]:.3f}")
        print(f"  {b:>12} -> {a:<14} {T[b][a]:.3f}")
off = [T[r][c] for r in POLARITY for c in POLARITY
       if r != c and not {r, c} in [{"cities", "neg_cities"},
                                    {"larger_than", "smaller_than"}]]
print(f"\n4x4 cross-family cells (excl. negation pairs): "
      f"{min(off):.3f} - {max(off):.3f}")
print("\nfull-matrix cells the 4x4 hides (fit on cities) -- NOT chance:")
for c, v in T["cities"].items():
    if c not in POLARITY:
        print(f"  cities -> {c:<26} {v:.3f}")
