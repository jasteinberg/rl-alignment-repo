"""
Figures for the steering sections.

'A causal test of the mass-mean direction' (the sign flip of chi) and 'Steering
without a rogue dimension' (the regime control), from artifacts/steer_ckpt/ (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).

Subcommands, with the script each replaces:

    steering-signflip      fig_steering_signflip.py
    regime-control         fig_regime_control.py
    all                    every figure above

Run from the repo root:  python scripts/figures/causal_steering.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.truthlib.estimators import chi_origin
from utils.truthlib.steering import load_null_cell, load_seed_cells

# ---- shared by several subcommands ------------------------------------------
MODEL = "pythia-2.8b"
ALPHAS = [0.5, 1.0, 2.0, 4.0]
C_PLAIN = "#c0392b"     # wrong-signed: red
C_WHIT = "#2c6fbb"      # corrected: blue


# =============================================================================
# steering-signflip  (was scripts/fig_steering_signflip.py)
# =============================================================================
SIGNFLIP_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_steering_signflip.png"


DATASET = "counterfact_true_false"


LAYERS = [20, 24, 28]


PANEL_A_LAYER = 28


CHI_AGG = os.environ.get("CHI_AGG", "median").lower()   # "median" | "mean"


C_NULL = "#7f8c8d"      # null band: grey


def arm_curve(seeds, arm):
    """Mean and sem of antisym(alpha) across seeds for one arm."""
    M = np.array([[s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS] for s in seeds])
    return M.mean(0), M.std(0, ddof=1) / np.sqrt(len(seeds))


def chi_across_seeds(seeds, arm):
    """Central chi across seeds, plus asymmetric error bars.

    CHI_AGG=median (default) reports the median with inter-quartile bars, matching
    what the post quotes and what chi_whitening_analysis.py computes. CHI_AGG=mean
    reports the mean with +/- s.e.m. The two differ materially on the plain arm,
    whose seed distribution is skewed: at L24 the median is -0.026 and the mean
    -0.018.
    """
    chis = [chi_origin(ALPHAS, [s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS]) for s in seeds]
    chis = np.asarray(chis, float)
    if CHI_AGG == "mean":
        mu = float(np.mean(chis))
        e = float(np.std(chis, ddof=1) / np.sqrt(len(chis)))
        return mu, (e, e)
    med = float(np.median(chis))
    q25, q75 = np.percentile(chis, [25, 75])
    return med, (float(med - q25), float(q75 - med))


def run_steering_signflip(argv=None):
    """Figure: wrong-signed plain steering vs. sign-corrected whitened steering on
    counterfact_true_false / pythia-2.8b. Reads existing artifacts/steer_ckpt/*.json
    only -- NO model, NO GPU.

    Panel A: antisymmetric response A(h), plain vs whitened, at L28, with the
             random-direction null band (mean +/- p95 envelope) shaded.
    Panel B: steering susceptibility chi = dA/dh|_0 across depth (L20/24/28),
             plain vs whitened, with the sign-flip made visible by the zero line.

    The steering field is written h in the post; ALPHAS and the "alphas" JSON keys are
    the on-disk data schema and are left as-is.
    """
    argparse.ArgumentParser(description=run_steering_signflip.__doc__).parse_args(argv)
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
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---------- Panel A: A(alpha) at PANEL_A_LAYER ----------
    seeds = load_seed_cells(MODEL, DATASET, PANEL_A_LAYER)
    null = load_null_cell(MODEL, DATASET, PANEL_A_LAYER)
    a = np.array(ALPHAS)

    # null band: signed 5th-95th percentiles of the null draws, per alpha. A point outside
    # the band in its own direction is a one-sided rank test at 5% -- the same criterion
    # as the quoted p-values (was +/- p95 of |A|, i.e. a one-sided 2.5% bar).
    null_lo = np.array([np.percentile(null["alphas"][str(al)]["draws"], 5) for al in ALPHAS])
    null_hi = np.array([np.percentile(null["alphas"][str(al)]["draws"], 95) for al in ALPHAS])
    axA.fill_between(a, null_lo, null_hi, color=C_NULL, alpha=0.18,
                     label="random-direction null (5th–95th pct.)", zorder=1)

    for arm, c, lab in [("plain", C_PLAIN, r"plain $\hat\theta\propto\hat\delta$"),
                        ("whitened", C_WHIT, r"whitened $\hat\theta_F\propto\hat\Sigma^{-1}\hat\delta$")]:
        m, e = arm_curve(seeds, arm)
        axA.errorbar(a, m, yerr=e, color=c, marker="o", ms=4, lw=1.6, capsize=2,
                     label=lab, zorder=3)

    axA.axhline(0, color="k", lw=0.7, zorder=2)
    axA.set_xlabel(r"steering field $h$  (class-gap units)")
    axA.set_ylabel(r"antisymmetric response $A(h)$")
    axA.set_title(rf"$L={PANEL_A_LAYER}$: plain steers the wrong way", fontsize=11)
    axA.legend(fontsize=8.5, frameon=False, loc="upper left")

    # ---------- Panel B: chi across depth ----------
    x = np.arange(len(LAYERS))
    w = 0.34
    for i, (arm, c, lab) in enumerate([("plain", C_PLAIN, "plain"),
                                        ("whitened", C_WHIT, "whitened")]):
        cent, lo, hi = [], [], []
        for L in LAYERS:
            sd = load_seed_cells(MODEL, DATASET, L)
            mu, (elo, ehi) = chi_across_seeds(sd, arm)
            cent.append(mu); lo.append(elo); hi.append(ehi)
        axB.bar(x + (i - 0.5) * w, cent, w, yerr=[lo, hi], color=c, alpha=0.85,
                capsize=3, label=lab)

    axB.axhline(0, color="k", lw=0.7)
    axB.set_xticks(x)
    axB.set_xticklabels([f"L{L}" for L in LAYERS])
    axB.set_ylabel(r"steering susceptibility $\chi=\mathrm{d}A/\mathrm{d}h|_0$")
    axB.set_title("rank-1 whitening flips the sign at depth"
                  + ("  (median, IQR)" if CHI_AGG == "median" else "  (mean, s.e.m.)"),
                  fontsize=11)
    axB.legend(fontsize=9, frameon=False)

    fig.suptitle(
        r"counterfact / pythia-2.8b: a rank-one correction converts wrong-signed steering to correct-signed",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    SIGNFLIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SIGNFLIP_OUT, bbox_inches="tight")
    print(f"wrote {SIGNFLIP_OUT}")

    # also dump the numbers the caption will quote, so they can be checked by eye
    print("\n-- caption numbers --")
    for L in LAYERS:
        sd = load_seed_cells(MODEL, DATASET, L)
        pm, _ = chi_across_seeds(sd, "plain")
        wm, _ = chi_across_seeds(sd, "whitened")
        npos_p = sum(1 for s in sd if s["alphas"]["1.0"]["plain"]["antisym"] > 0)
        npos_w = sum(1 for s in sd if s["alphas"]["1.0"]["whitened"]["antisym"] > 0)
        print(f"L{L}: chi_plain={pm:+.4f} chi_whit={wm:+.4f} "
              f"plain +seeds={npos_p}/{len(sd)} whit +seeds={npos_w}/{len(sd)}")


# =============================================================================
# regime-control  (was scripts/fig_regime_control.py)
# =============================================================================
REGIME_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_regime_control.png"


PANELS = [
    ("cities", "cities: mass-mean steers, whitening degrades", [8, 12, 16, 20, 24, 28]),
    ("counterfact_true_false", "counterfact: the assignment inverts", [20, 24, 28]),
]


def chi_stats(seeds, arm):
    c = np.array([chi_origin(ALPHAS, [s["alphas"][str(a)][arm]["antisym"] for a in ALPHAS])
                  for s in seeds], float)
    med = float(np.median(c))
    q25, q75 = np.percentile(c, [25, 75])
    return med, max(med - q25, 0.0), max(q75 - med, 0.0)


def run_regime_control(argv=None):
    """Figure: the rogue dimension decides which regime a dataset is in.

    Same model (pythia-2.8b), same estimator, same steering protocol -- two datasets.
    On cities the plain difference-in-means direction steers correctly and whitening
    degrades it, exactly as mass-mean probing intends. On counterfact_true_false at
    depth the assignment inverts: plain is significantly wrong-signed and the whitened
    direction is the one that steers. Reads existing artifacts/steer_ckpt/*.json only
    -- NO model, NO GPU.

    chi is the median across seeds (see fig_steering_signflip.py); bars are IQR.
    """
    argparse.ArgumentParser(description=run_regime_control.__doc__).parse_args(argv)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)

    for ax, (dataset, title, layers) in zip(axes, PANELS):
        have = [L for L in layers if load_seed_cells(MODEL, dataset, L)]
        x = np.arange(len(have))
        w = 0.34
        for i, (arm, c, lab) in enumerate([("plain", C_PLAIN, "plain"),
                                           ("whitened", C_WHIT, "whitened")]):
            cent, lo, hi = [], [], []
            for L in have:
                m, a, b = chi_stats(load_seed_cells(MODEL, dataset, L), arm)
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
        "pythia-2.8b: the causal direction is set by the presence or absence of a rogue dimension",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    REGIME_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REGIME_OUT, bbox_inches="tight")
    print(f"wrote {REGIME_OUT}")

    for dataset, _, layers in PANELS:
        for L in layers:
            sd = load_seed_cells(MODEL, dataset, L)
            if not sd:
                continue
            p, _, _ = chi_stats(sd, "plain")
            w_, _, _ = chi_stats(sd, "whitened")
            npos = sum(1 for s in sd if s["alphas"]["1.0"]["plain"]["antisym"] > 0)
            print(f"{dataset:>24} L{L:<3} n={len(sd):<2} chi_plain={p:+.4f} "
                  f"chi_whit={w_:+.4f}  plain +{npos}/{len(sd)}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_steering_signflip, run_regime_control,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "steering-signflip": run_steering_signflip,
    "regime-control": run_regime_control,
    "all": run_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    cmd = sys.argv[1]
    sys.argv[0] = f"{Path(sys.argv[0]).name} {cmd}"      # argparse usage names the subcommand
    COMMANDS[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
