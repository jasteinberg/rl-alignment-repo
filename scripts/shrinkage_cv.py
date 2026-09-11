"""
Choose the shrinkage intensity by cross-validation INSIDE the training half.

scripts/shrinkage_sweep.py showed that Ledoit-Wolf's rho is far from optimal for the
separation d'_F of the direction it produces -- but it found that by reading held-out d'
off a grid, which is selection on the evaluation data and therefore not quotable. This
does it properly:

    inner loop   K-fold CV within the TRAIN half only; for each rho, fit
                 theta_F on K-1 folds and score d' on the held-out fold;
                 rho* = argmax of the mean CV score.
    outer step   refit on the FULL train half at rho*, score ONCE on the
                 test half. The test half is touched exactly once per layer.

Ledoit-Wolf minimizes E||Sigma_hat - Sigma||_F^2. That is not the objective anyone
cares about here: the quantity that matters is d' of Sigma_hat^-1 delta_hat, which
depends on the inverse and on one particular direction in it. CV optimizes the thing
we actually report.

Implementation note: Sigma(rho) = (1-rho) C + rho mu I shares eigenvectors with C, so
one eigendecomposition per fold serves the whole rho grid --
    Sigma(rho)^-1 delta = V diag(1/((1-rho) lambda + rho mu)) V^T delta.
Without this the sweep costs a 2560x2560 solve per (fold, rho).

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf
from truthlib.estimators import split_indices, within_class_center, d_prime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "shrinkage_cv.json")
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 0.9]


def eig_fit(X, y):
    """Eigendecompose the within-class covariance once; return pieces for any rho."""
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)
    mu = float(lam.mean())
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    return lam, V, mu, dlt


def theta_at(lam, V, mu, dlt, rho):
    w = V.T @ dlt
    t = V @ (w / ((1.0 - rho) * lam + rho * mu))
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def folds_for(y, K, rng):
    """Class-stratified K folds."""
    out = [[] for _ in range(K)]
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        for f, chunk in enumerate(np.array_split(idx, K)):
            out[f].append(chunk)
    return [np.concatenate(c) for c in out]


def run_layer(X, y, tr, te, K, seed, n_null=200):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    rng = np.random.default_rng(seed + 1)

    # ---- inner CV, train half only -------------------------------------
    fs = folds_for(ytr, K, rng)
    scores = {r: [] for r in GRID}
    for f in range(K):
        va = fs[f]
        fit = np.concatenate([fs[g] for g in range(K) if g != f])
        if len(np.unique(ytr[fit])) < 2 or len(np.unique(ytr[va])) < 2:
            continue
        lam, V, mu, dlt = eig_fit(Xtr[fit], ytr[fit])
        for r in GRID:
            t = theta_at(lam, V, mu, dlt, r)
            scores[r].append(d_prime(Xtr[va] @ t, ytr[va]))
    cv = {r: float(np.mean(v)) for r, v in scores.items() if v}
    rho_star = max(cv, key=cv.get)

    # ---- one scoring pass on the test half -----------------------------
    lam, V, mu, dlt = eig_fit(Xtr, ytr)
    rho_lw = float(LedoitWolf(assume_centered=True)
                   .fit(within_class_center(Xtr, ytr)).shrinkage_)

    def scored(r):
        t = theta_at(lam, V, mu, dlt, r)
        if (Xtr @ t)[ytr == 1].mean() < (Xtr @ t)[ytr == 0].mean():
            t = -t                                   # orient on train only
        return d_prime(Xte @ t, yte)

    nulls = []
    rngn = np.random.default_rng(seed)
    for _ in range(n_null):
        u = rngn.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
        nulls.append(d_prime(Xte @ u, yte))

    return {"rho_star": rho_star, "rho_LW": rho_lw,
            "d_F_cv": scored(rho_star), "d_F_LW": scored(rho_lw),
            "cv_curve": {str(k): v for k, v in cv.items()},
            "null_p95": float(np.percentile(nulls, 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    results = {}
    for model in ("pythia-2.8b", "pythia-1.4b"):
        for ds in ("counterfact_true_false", "cities"):
            p = os.path.join(CACHE, f"{model}__{ds}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p); y = z["y"]
            tr, te = split_indices(y, seed=args.seed)
            layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
            key = f"{model}__{ds}"
            print(f"\n=== {key} ===", flush=True)
            print(f"{'L':>4} {'rho*':>8} {'rho_LW':>8} {'d_F(cv)':>8} {'d_F(LW)':>8} "
                  f"{'null':>7} {'ratio':>6}", flush=True)
            per = {}
            for L in layers:
                r = run_layer(z[f"L{L}"].astype(np.float64), y, tr, te,
                              args.folds, args.seed)
                per[str(L)] = r
                ratio = r["d_F_cv"] / r["d_F_LW"] if r["d_F_LW"] > 0 else float("inf")
                print(f"{L:>4} {r['rho_star']:>8.4g} {r['rho_LW']:>8.4f} "
                      f"{r['d_F_cv']:>8.3f} {r['d_F_LW']:>8.3f} "
                      f"{r['null_p95']:>7.3f} {ratio:>6.1f}", flush=True)
            results[key] = per

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "folds": args.folds, "grid": GRID,
                              "protocol": "rho chosen by K-fold CV inside the train "
                                          "half; test half scored once per layer"},
                   "results": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
