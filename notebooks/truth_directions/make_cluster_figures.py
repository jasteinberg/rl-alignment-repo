"""
notebooks/truth_directions/make_cluster_figures.py

Cluster geometry figures for the truth-directions post.
Real activations (cached, no model needed) + one schematic.
Gridlines on every axis.

Drafted with the assistance of Claude (Anthropic).
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/public-repos/rl-alignment-repo/artifacts/act_cache"
OUT = f"{HOME}/public-repos/website/assets/figures"
os.makedirs(OUT, exist_ok=True)

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


# ---------- Figure 1: real cluster geometry, cities vs counterfact ----------
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
    fig.savefig(f"{OUT}/truth_clusters.png", dpi=200)
    plt.close(fig)
    print("saved truth_clusters.png")


# ---------- Figure 2: schematic, mass-mean vs Fisher under anisotropy ----------
def fig_whitening_schematic():
    rng = np.random.default_rng(0)
    n = 500
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, (title, S) in zip(axes, [
            ("isotropic noise:  $\\hat\\theta$ is already optimal", np.diag([1.0, 1.0])),
            ("anisotropic noise:  $\\hat\\theta$ is not", np.diag([9.0, 0.35]))]):
        d = np.array([1.4, 1.4])                       # class-mean gap
        A = np.linalg.cholesky(S)
        X0 = rng.standard_normal((n, 2)) @ A.T - d / 2
        X1 = rng.standard_normal((n, 2)) @ A.T + d / 2
        ax.scatter(X0[:, 0], X0[:, 1], s=6, alpha=0.35, c=C_FALSE, label="false")
        ax.scatter(X1[:, 0], X1[:, 1], s=6, alpha=0.35, c=C_TRUE, label="true")

        th = d / np.linalg.norm(d)                     # mass-mean
        thf = np.linalg.solve(S, d); thf /= np.linalg.norm(thf)   # Fisher

        for v, c, lab in ((th, "#222", r"$\hat\theta \propto \delta$"),
                          (thf, "#e0a106", r"$\hat\theta_F \propto \Sigma^{-1}\delta$")):
            ax.annotate("", xy=3.2 * v, xytext=-3.2 * v,
                        arrowprops=dict(arrowstyle="-", lw=2.2, color=c))
            ax.plot([], [], color=c, lw=2.2, label=lab)

        def dp(u):
            p1, p0 = X1 @ u, X0 @ u
            return abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var() + p0.var()))
        ax.set_title(f"{title}\n" + rf"$d'(\hat\theta)={dp(th):.2f}$,  $d'(\hat\theta_F)={dp(thf):.2f}$")
        ax.set_xlim(-7, 7); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="upper left", markerscale=2); grid(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT}/truth_whitening_schematic.png", dpi=200)
    plt.close(fig)
    print("saved truth_whitening_schematic.png")


# ---------- Figure 3: the decoding null, drawn not asserted ----------
def fig_null_schematic():
    z = np.load(f"{CACHE}/pythia-2.8b__cities.npz")
    y = z["y"]; X = z[f"L{LAYER}"].astype(np.float64)
    th, w, V, dp_true = geometry(X, y)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(400):
        u = rng.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
        p = X @ u; p1, p0 = p[y == 1], p[y == 0]
        ds.append(abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1))))
    ds = np.array(ds); p95 = np.percentile(ds, 95)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(ds, bins=40, color="#8a8a8a", alpha=0.75, label="400 random directions")
    ax.axvline(p95, color="#444", ls="--", lw=1.6, label=rf"null $p_{{95}}$ = {p95:.2f}")
    ax.axvline(dp_true, color=C_TRUE, lw=2.4, label=rf"$\hat\theta$: $d'$ = {dp_true:.2f}")
    ax.set_xlabel(r"$d'(u)$"); ax.set_ylabel("count")
    ax.set_title(f"The decoding null: cities, pythia-2.8b layer {LAYER}")
    ax.legend(fontsize=9); grid(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/truth_null_distribution.png", dpi=200)
    plt.close(fig)
    print("saved truth_null_distribution.png")


if __name__ == "__main__":
    fig_clusters()
    fig_whitening_schematic()
    fig_null_schematic()
