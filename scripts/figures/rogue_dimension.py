"""
Figures for 'The rogue dimension'.

Cluster geometry on real activations, the depth arms (theta, v1, theta_perp), the
superposition probe, and the lifetime of the rogue dimension against the massive
coordinates (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).

Subcommands, with the script each replaces:

    clusters               fig_clusters.py
    depth-arms             fig_depth_arms.py
    superposition          fig_superposition.py
    rogue-lifetime         fig_rogue_lifetime.py
    all                    every figure above

Run from the repo root:  python scripts/figures/rogue_dimension.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
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


# ---- shared by several subcommands ------------------------------------------
SWEEP = Path(os.environ.get("SNR_SWEEP", REPO / "artifacts" / "snr_sweep.json"))
S = json.load(open(SWEEP))


# =============================================================================
# clusters  (was scripts/fig_clusters.py)
# =============================================================================
CACHE = f"{REPO}/artifacts/act_cache"


CLUSTERS_OUT = str(Path(os.environ.get("FIGURE_DIR", REPO / "figures")))


C_TRUE, C_FALSE = "#1f6f8b", "#c1553b"


LAYER = 28


def grid(ax):
    ax.grid(True, ls=":", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)


def geometry(X, y):
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] -= X[y == lab].mean(0)
    C = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    p = X @ th
    p1, p0 = p[y == 1], p[y == 0]
    dprime = abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1)))
    return th, w, V, dprime


def fig_clusters():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for r, ds in enumerate(["cities", "counterfact_true_false"]):
        z = np.load(f"{CACHE}/pythia-2.8b__{ds}.npz")
        y = z["y"]
        X = z[f"L{LAYER}"].astype(np.float64)
        th, w, V, dp = geometry(X, y)
        v1, v2 = V[:, 0], V[:, 1]
        cos_tv1 = abs(th @ v1)

        # (a) top two principal axes
        ax = axes[r, 0]
        a1, a2 = X @ v1, X @ v2
        for lab, c, m in ((1, C_TRUE, "true"), (0, C_FALSE, "false")):
            ax.scatter(a1[y == lab], a2[y == lab], s=5, alpha=0.45, c=c, label=m)
        ax.set_xlabel(r"$x\cdot v_1$  (top principal axis)")
        ax.set_ylabel(r"$x\cdot v_2$")
        ax.set_title(f"{ds}\n" + rf"$\lambda_1/\mathrm{{tr}}\,\Sigma$ = {w[0]/w.sum():.3f}")
        ax.legend(markerscale=2, fontsize=8, loc="best"); grid(ax)

        # (b) projection onto the mass-mean direction
        ax = axes[r, 1]
        p = X @ th
        bins = np.linspace(p.min(), p.max(), 55)
        ax.hist(p[y == 1], bins=bins, alpha=0.6, color=C_TRUE, label="true")
        ax.hist(p[y == 0], bins=bins, alpha=0.6, color=C_FALSE, label="false")
        ax.set_xlabel(r"$x\cdot\hat\theta$")
        ax.set_ylabel("count")
        ax.set_title(rf"$d' = {dp:.2f}$,   $|\cos(\hat\theta, v_1)| = {cos_tv1:.3f}$")
        ax.legend(fontsize=8); grid(ax)

        # (c) eigenvalue spectrum
        ax = axes[r, 2]
        k = 40
        ax.semilogy(np.arange(1, k + 1), w[:k], "o-", ms=3, lw=1.2, color="#444")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel(r"$\lambda_i$")
        ax.set_title(rf"$\lambda_1/\lambda_2 = {w[0]/w[1]:.3g}$")
        grid(ax)

    fig.suptitle(f"Within-class geometry, pythia-2.8b layer {LAYER}", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{CLUSTERS_OUT}/truth_clusters.png", dpi=200)
    plt.close(fig)
    print("saved truth_clusters.png")


def run_clusters(argv=None):
    """Within-class cluster geometry at pythia-2.8b layer 28, cities vs counterfact: top principal
    axes, the mass-mean projection, and the eigenvalue spectrum (truth_clusters.png).

    Split out of notebooks/truth_directions/make_cluster_figures.py; reads this repo's
    act cache and writes to $FIGURE_DIR (default figures/), like the other figure scripts.
    """
    argparse.ArgumentParser(description=run_clusters.__doc__).parse_args(argv)
    os.makedirs(CLUSTERS_OUT, exist_ok=True)
    fig_clusters()


# =============================================================================
# depth-arms  (was scripts/fig_depth_arms.py)
# =============================================================================
ARMS_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_depth_arms.png"


ARMS_MODEL = "EleutherAI/pythia-2.8b"


PANELS = [("counterfact_true_false", "`counterfact`"), ("cities", "`cities`")]


C_PLAIN, C_WHIT, C_PERP = "#c0392b", "#2E6DA4", "#d68910"


ROGUE = Path(os.environ.get("ROGUE_JSON", REPO / "artifacts" / "rogue_dimension.json"))


R = json.load(open(ROGUE))


def perp(dataset):
    node = R[dataset]
    Ls = sorted(node, key=int)
    return (np.array([float(L) for L in Ls]),
            np.array([node[L]["auroc"]["theta_perp"] for L in Ls]))


def arms(dataset):
    node = S["models"][ARMS_MODEL]["datasets"][dataset]
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


def run_depth_arms(argv=None):
    """Depth profiles of the two estimators against the decoding null.

    Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

    Held-out AUROC per layer for the mass-mean and whitened directions, with the
    per-layer random-direction null shaded from 1/2 up to its 95th percentile. A
    curve inside the band is not recoverable, and height above the band is the
    margin m = AUROC - p95 that the layer-selection rule maximises, so level and
    clearance are readable from the same axes.
    """
    argparse.ArgumentParser(description=run_depth_arms.__doc__).parse_args(argv)
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
    ARMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARMS_OUT, bbox_inches="tight")
    print(f"wrote {ARMS_OUT}")

    for ds, label in PANELS:
        L, plain, whit, null, best = arms(ds)
        print(f"{ds}: best_layer L{best}; plain clears at "
              f"{first_clear(L, plain, null)}, whitened at {first_clear(L, whit, null)}")
        for i in [26, 28, 31, 32]:
            j = int(np.where(L == i)[0][0])
            print(f"   L{i}: plain {plain[j]:.3f}  whit {whit[j]:.3f}  null {null[j]:.3f}")


# =============================================================================
# superposition  (was scripts/fig_superposition.py)
# =============================================================================
SUPER_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_superposition.png"


SUPER_MODEL = "EleutherAI/pythia-2.8b"


SERIES = [
    ("cities", "cities", "#7f8c8d", False),
    ("neg_cities", "neg_cities", "#95a5a6", False),
    ("larger_than", "larger_than", "#aab3b5", False),
    ("counterfact_true_false", "counterfact", "#c0392b", True),
]


def run_superposition(argv=None):
    """Figure: the superposition probe -- d' of the mass-mean direction after projecting
    out the top-k principal components of the activations. Reads the existing
    artifacts/snr_sweep.json only -- NO model, NO GPU.

    Reads as a salience knob. If a truth direction merely rides the high-variance
    subspace, removing the leading components destroys it; if it occupies its own
    low-variance subspace, d' survives. counterfact_true_false does neither: d' RISES
    as the top directions are stripped, because the leading eigendirection is the
    massive-activation axis the mass-mean estimator has collapsed onto -- the
    rogue-dimension diagnosis, visible here before it is named.
    """
    argparse.ArgumentParser(description=run_superposition.__doc__).parse_args(argv)
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
    node = S["models"][SUPER_MODEL]["datasets"]

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
    SUPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPER_OUT, bbox_inches="tight")
    print(f"wrote {SUPER_OUT}")

    for key, label, _, _ in SERIES:
        sp = node[key]["superposition"]
        row = "  ".join(f"k={r['k']}:{r['d_prime']:.2f}" for r in sp)
        print(f"{label:>14}  {row}")


# =============================================================================
# rogue-lifetime  (was scripts/fig_rogue_lifetime.py)
# =============================================================================
GEOM = REPO / "artifacts" / "geometry_all_layers.json"


MASS = REPO / "artifacts" / "massive_all_layers.json"


LIFETIME_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_rogue_lifetime.png"


LIFETIME_MODEL = "pythia-2.8b"


C_CF, C_CIT, C_SPAN = "#0f9b8e", "#c0392b", "#7f8c8d"


G = json.load(open(GEOM))


M = json.load(open(MASS))


def series(src, ds, field):
    rows = src[f"{LIFETIME_MODEL}__{ds}"]["layers"]
    ls = sorted(int(k) for k in rows)
    xs, ys = [], []
    for L in ls:
        r = rows[str(L)]
        if r.get("degenerate") or field not in r:
            continue
        xs.append(L)
        ys.append(r[field])
    return np.array(xs), np.array(ys)


def massive_span(ds):
    """First and last layer at which any coordinate qualifies as massive."""
    rows = M[f"{LIFETIME_MODEL}__{ds}"]["layers"]
    live = [int(k) for k, r in rows.items() if r["n_massive_coords"] > 0]
    return min(live), max(live)


def run_rogue_lifetime(argv=None):
    """Figure: the lifetime of the rogue dimension across depth, pythia-2.8b.

    Two panels sharing the depth axis, one measured quantity each (no second scale):
      top     |cos(theta_hat, v1)|, how far the estimator has collapsed onto the axis
      bottom  PR, the spectrum condition, log scale

    The shaded span marks the layers where the massive activation is present, by the
    massive_mask criterion of the eleven-outlier appendix. It is the same span on
    both datasets, because the same coordinates qualify on both. What differs is
    droppers: eleven on `counterfact` at every layer of the span, zero on `cities`.

    Reads artifacts/geometry_all_layers.json and artifacts/massive_all_layers.json
    only -- NO model, NO GPU.
    """
    argparse.ArgumentParser(description=run_rogue_lifetime.__doc__).parse_args(argv)
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
    lo, hi = massive_span("counterfact_true_false")
    assert (lo, hi) == massive_span("cities"), "span should be identical on both sets"

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)

    for ax, field, label in [
        (axes[0], "cos_theta_v1", r"$|\cos(\hat\theta,\, \hat v_1)|$"),
        (axes[1], "PR", "participation ratio"),
    ]:
        ax.axvspan(lo - 0.5, hi + 0.5, color=C_SPAN, alpha=0.11, zorder=0, lw=0)
        for ds, c, name in [("counterfact_true_false", C_CF, "counterfact"),
                            ("cities", C_CIT, "cities")]:
            x, y = series(G, ds, field)
            ax.plot(x, y, lw=2.0, color=c, zorder=3, label=name)
        ax.set_ylabel(label)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("layer")
    axes[0].set_ylim(-0.03, 1.22)
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left",
                   bbox_to_anchor=(0.06, 0.02))

    # the massive activation dies at hi+1; name it once, on the top panel
    axes[0].annotate("massive activation gone",
                     xy=(hi + 1.2, 0.50), xytext=(hi - 8.0, 0.72),
                     fontsize=9.5, color="#566573", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=0.9))
    axes[0].text((lo + hi) / 2, 1.19, "massive activation present",
                 ha="center", va="top", fontsize=9, color="#566573")

    fig.tight_layout()
    LIFETIME_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LIFETIME_OUT, bbox_inches="tight")
    print(f"wrote {LIFETIME_OUT}\n")

    print(f"massive activation present: layers {lo}-{hi} (both datasets)")
    for ds in ["counterfact_true_false", "cities"]:
        rows = M[f"{LIFETIME_MODEL}__{ds}"]["layers"]
        d = {int(k): r["n_droppers"] for k, r in rows.items()}
        print(f"  {ds:<22} droppers in span: "
              f"{sorted(set(d[L] for L in range(lo, hi + 1)))}")
        x, y = series(G, ds, "cos_theta_v1")
        for L in (hi - 1, hi, hi + 1, hi + 2):
            if L in set(x):
                j = list(x).index(L)
                print(f"    L{L:>2}  cos {y[j]:.3f}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_clusters, run_depth_arms, run_superposition, run_rogue_lifetime,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "clusters": run_clusters,
    "depth-arms": run_depth_arms,
    "superposition": run_superposition,
    "rogue-lifetime": run_rogue_lifetime,
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
