#!/usr/bin/env python
"""
Figure: wrong-signed plain steering vs. sign-corrected whitened steering on
counterfact_true_false / pythia-2.8b. Reads existing artifacts/steer_ckpt/*.json
only -- NO model, NO GPU.

Panel A: antisymmetric response A(alpha), plain vs whitened, at L28, with the
         random-direction null band (mean +/- p95 envelope) shaded.
Panel B: steering susceptibility chi = dA/dalpha|_0 across depth (L20/24/28),
         plain vs whitened, with the sign-flip made visible by the zero line.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("STEER_CKPT", REPO / "artifacts" / "steer_ckpt"))
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_steering_signflip.png"
MODEL = "pythia-2.8b"
DATASET = "counterfact_true_false"
ALPHAS = [0.5, 1.0, 2.0, 4.0]
LAYERS = [20, 24, 28]
PANEL_A_LAYER = 28

# physics-paper-ish styling: serif, restrained palette
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
C_PLAIN = "#c0392b"     # wrong-signed: red
C_WHIT = "#2c6fbb"      # corrected: blue
C_NULL = "#7f8c8d"      # null band: grey


def load_seeds(layer):
    seeds = []
    for f in sorted(CKPT.glob(f"{MODEL}__{DATASET}__L{layer}__s*.json")):
        seeds.append(json.load(open(f)))
    return seeds


def load_null(layer):
    return json.load(open(CKPT / f"{MODEL}__{DATASET}__L{layer}__NULL.json"))


def arm_curve(seeds, arm):
    """Mean and sem of antisym(alpha) across seeds for one arm."""
    M = np.array([[s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS] for s in seeds])
    return M.mean(0), M.std(0, ddof=1) / np.sqrt(len(seeds))


def chi_origin(vals):
    a = np.array(ALPHAS, float)
    v = np.array(vals, float)
    return float((a * v).sum() / (a * a).sum())


def chi_across_seeds(seeds, arm):
    chis = [chi_origin([s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS]) for s in seeds]
    return np.mean(chis), np.std(chis, ddof=1) / np.sqrt(len(chis))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

# ---------- Panel A: A(alpha) at PANEL_A_LAYER ----------
seeds = load_seeds(PANEL_A_LAYER)
null = load_null(PANEL_A_LAYER)
a = np.array(ALPHAS)

# null band: mean +/- p95-implied envelope, per alpha (folded to symmetric grey band)
null_mean = np.array([null["alphas"][str(al)]["antisym_mean"] for al in ALPHAS])
null_p95 = np.array([null["alphas"][str(al)]["antisym_p95"] for al in ALPHAS])
axA.fill_between(a, -null_p95, null_p95, color=C_NULL, alpha=0.18,
                 label="random-direction null (95%)", zorder=1)

for arm, c, lab in [("plain", C_PLAIN, r"plain $\hat\theta\propto\hat\delta$"),
                    ("whitened", C_WHIT, r"whitened $\hat\theta_F\propto\hat\Sigma^{-1}\hat\delta$")]:
    m, e = arm_curve(seeds, arm)
    axA.errorbar(a, m, yerr=e, color=c, marker="o", ms=4, lw=1.6, capsize=2,
                 label=lab, zorder=3)

axA.axhline(0, color="k", lw=0.7, zorder=2)
axA.set_xlabel(r"steering scale $\alpha$  (class-gap units)")
axA.set_ylabel(r"antisymmetric response $A(\alpha)$")
axA.set_title(rf"$L={PANEL_A_LAYER}$: plain steers the wrong way", fontsize=11)
axA.legend(fontsize=8.5, frameon=False, loc="upper left")

# ---------- Panel B: chi across depth ----------
x = np.arange(len(LAYERS))
w = 0.34
for i, (arm, c, lab) in enumerate([("plain", C_PLAIN, "plain"),
                                    ("whitened", C_WHIT, "whitened")]):
    means, errs = [], []
    for L in LAYERS:
        sd = load_seeds(L)
        mu, se = chi_across_seeds(sd, arm)
        means.append(mu); errs.append(se)
    axB.bar(x + (i - 0.5) * w, means, w, yerr=errs, color=c, alpha=0.85,
            capsize=3, label=lab)

axB.axhline(0, color="k", lw=0.7)
axB.set_xticks(x)
axB.set_xticklabels([f"L{L}" for L in LAYERS])
axB.set_ylabel(r"steering susceptibility $\chi=\mathrm{d}A/\mathrm{d}\alpha|_0$")
axB.set_title("rank-1 whitening flips the sign at depth", fontsize=11)
axB.legend(fontsize=9, frameon=False)

fig.suptitle(
    r"counterfact / pythia-2.8b: a rank-one correction converts wrong-signed steering to correct-signed",
    fontsize=11.5, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

# also dump the numbers the caption will quote, so they can be checked by eye
print("\n-- caption numbers --")
for L in LAYERS:
    sd = load_seeds(L)
    pm, _ = chi_across_seeds(sd, "plain")
    wm, _ = chi_across_seeds(sd, "whitened")
    npos_p = sum(1 for s in sd if s["alphas"]["1.0"]["plain"]["antisym"] > 0)
    npos_w = sum(1 for s in sd if s["alphas"]["1.0"]["whitened"]["antisym"] > 0)
    print(f"L{L}: chi_plain={pm:+.4f} chi_whit={wm:+.4f} "
          f"plain +seeds={npos_p}/{len(sd)} whit +seeds={npos_w}/{len(sd)}")
