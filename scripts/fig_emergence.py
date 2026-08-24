"""
Figure: emergence of the truth direction across the Pythia scale ladder --
best-layer held-out AUROC of the mass-mean direction against the
random-direction null, 70m -> 2.8b.

Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

Selection: the artifact's own `best_layer` (max held-out AUROC above that layer's
own null). Both arms are drawn, because the claim is not "AUROC rises" but
"nothing in this linear class clears the null until 410m" -- a statement about
the whitened arm too.

Supersedes an ad-hoc Jul-8 figure that had no generator and whose numbers no
longer matched the Jul-9 sweep.

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
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_emergence.png"

DATASET = os.environ.get("EMERGENCE_DS", "cities")
LADDER = [
    ("EleutherAI/pythia-70m", "70m"),
    ("EleutherAI/pythia-410m", "410m"),
    ("EleutherAI/pythia-1.4b", "1.4b"),
    ("EleutherAI/pythia-2.8b", "2.8b"),
]

C_PLAIN, C_WHIT, C_NULL = "#c0392b", "#2c6fbb", "#7f8c8d"

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


def at_best_layer(model):
    node = S["models"][model]["datasets"][DATASET]
    rec = next(r for r in node["layers"] if r["layer"] == node["best_layer"])
    return rec, node["d_model"], node["n"]


rows = []
for model, label in LADDER:
    rec, d_model, n = at_best_layer(model)
    rows.append({
        "label": label,
        "d_model": d_model,
        "n": n,
        "layer": rec["layer"],
        "plain": rec["plain"]["auroc"],
        "plain_dp": rec["plain"]["d_prime"],
        "whit": rec["whitened"]["auroc"],
        "whit_dp": rec["whitened"]["d_prime"],
        "null": rec["null"]["auroc_p95"],
    })

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.fill_between(x, 0.5, [r["null"] for r in rows],
                color=C_NULL, alpha=0.18, zorder=1,
                label="random-direction null (to $p_{95}$)")
ax.plot(x, [r["null"] for r in rows], color=C_NULL, lw=1.0, ls="--", zorder=2)

ax.plot(x, [r["plain"] for r in rows], marker="o", ms=6, lw=2.2,
        color=C_PLAIN, zorder=4, label=r"mass-mean $\hat\theta$")
ax.plot(x, [r["whit"] for r in rows], marker="s", ms=5, lw=1.4,
        color=C_WHIT, zorder=3, label=r"whitened $\hat\theta_{\mathrm{F}}$")

ax.axhline(0.5, color="#bdc3c7", lw=0.8, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['label']}\n$d$={r['d_model']}" for r in rows])
ax.set_xlabel("Pythia model size")
ax.set_ylabel("held-out AUROC at best layer")
ax.set_ylim(0.35, 1.02)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title(f"Emergence of the truth direction ({DATASET})", fontsize=11, pad=10)

ax.annotate("inside the null:\nnothing recoverable",
            xy=(0, rows[0]["plain"]), xytext=(0.18, 0.66),
            fontsize=9.5, color=C_PLAIN,
            arrowprops=dict(arrowstyle="->", color=C_PLAIN, lw=1.0))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}\n")

hdr = f"{'model':>6} {'L':>3} {'plain':>7} {'d_prime':>8} {'whit':>7} {'null95':>7} {'margin':>8} {'clears?':>8}"
print(hdr)
for r in rows:
    m = r["plain"] - r["null"]
    print(f"{r['label']:>6} {r['layer']:>3} {r['plain']:>7.3f} {r['plain_dp']:>8.3f} "
          f"{r['whit']:>7.3f} {r['null']:>7.3f} {m:>+8.3f} {'yes' if m > 0 else 'NO':>8}")
print("\nwhitened arm vs null (does ANY linear correction clear it?):")
for r in rows:
    m = r["whit"] - r["null"]
    print(f"  {r['label']:>6} whit {r['whit']:.3f} vs null {r['null']:.3f} -> {m:+.3f} "
          f"{'clears' if m > 0 else 'INSIDE NULL'}")
