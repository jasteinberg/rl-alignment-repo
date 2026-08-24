"""
Decompose the Ledoit-Wolf shrinkage intensity and test whether heavy tails, not
concentration, drive it -- and compare train-half vs full-sample spectra.

Two questions.

(1) WHY IS rho LARGE FOR counterfact?  LW picks
        rho = min(beta2, delta2) / delta2,
        delta2 = (1/d) ||C - mu I||_F^2,
        beta2  = (1/(d N^2)) sum_i ||x_i x_i^T - C||_F^2
                = (1/(d N^2)) [ sum_i ||x_i||^4 - N ||C||_F^2 ].
    For Gaussian x, E sum_i ||x_i||^4 = N[(tr C)^2 + 2 tr C^2], which gives a
    Gaussian reference rho_gauss. In the rank-1 limit rho_gauss -> 2/N. Reporting
    rho_obs / rho_gauss isolates the non-Gaussian part; excess kurtosis of the
    projection onto v1 says whether the rogue axis is where it lives.

(2) WHICH SPLIT SHOULD THE SPECTRUM BE QUOTED ON?  The eigenvalues are descriptive
    statistics of Sigma, not a fitted predictor scored on itself, so the full
    sample is the better estimator -- PROVIDED any tuned quantity (rho) was chosen
    on train. But rho itself is N-dependent (beta2 ~ 1/N), so rho_train applied to
    a full-sample C over-shrinks. This reports rho at both N and the resulting
    lambda1/lambda2 under each choice, so the size of that effect is visible.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "shrinkage_decomposition.json")


def split_indices(y, frac=0.5, seed=0):
    """Class-stratified train/test split -- identical to snr_sweep.split_indices."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        tr.append(idx[:k]); te.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)


def within_class_center(X, y):
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] = X[y == lab] - X[y == lab].mean(0)
    return Xc


def lw_parts(Xc):
    """Reproduce sklearn's LW pieces explicitly so beta2/delta2 are inspectable."""
    N, d = Xc.shape
    C = (Xc.T @ Xc) / N
    mu = float(np.trace(C) / d)
    normF2 = float((C ** 2).sum())
    delta2 = (normF2 - d * mu ** 2) / d          # ||C - mu I||_F^2 / d
    sq = (Xc ** 2).sum(1)                        # ||x_i||^2
    beta2_obs = float((sq ** 2).sum() - N * normF2) / (d * N ** 2)
    rho_obs = min(beta2_obs, delta2) / delta2

    # Gaussian reference: E sum ||x||^4 = N[(tr C)^2 + 2 tr C^2]
    trC = float(np.trace(C))
    trC2 = normF2
    beta2_gauss = (trC ** 2 + 2 * trC2 - trC2) / (d * N)
    rho_gauss = min(beta2_gauss, delta2) / delta2
    return C, mu, delta2, beta2_obs, beta2_gauss, rho_obs, rho_gauss


def spectrum(C, rho, mu):
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    lam_s = (1.0 - rho) * lam + rho * mu
    r = lambda v: float(v[0] / v[1]) if v[1] > 0 else float("inf")
    return lam, r(lam), r(lam_s), float(lam.sum() ** 2 / (lam ** 2).sum())


def analyze(X, y, seed=0):
    tr, _ = split_indices(y, seed=seed)
    out = {}
    for tag, rows in (("train", tr), ("full", np.arange(len(y)))):
        Xc = within_class_center(X[rows], y[rows])
        C, mu, d2, b2o, b2g, rho_o, rho_g = lw_parts(Xc)
        chk = float(LedoitWolf(assume_centered=True).fit(Xc).shrinkage_)
        lam, r_raw, r_shr, PR = spectrum(C, rho_o, mu)

        # excess kurtosis of the within-class-centered projection onto v1
        v1 = np.linalg.eigh(C)[1][:, -1]
        p = Xc @ v1
        s = p.std()
        kurt_v1 = float(((p / s) ** 4).mean() - 3.0) if s > 0 else float("nan")
        # and of a typical bulk direction, for contrast
        vb = np.linalg.eigh(C)[1][:, -50]
        pb = Xc @ vb
        sb = pb.std()
        kurt_bulk = float(((pb / sb) ** 4).mean() - 3.0) if sb > 0 else float("nan")

        out[tag] = {
            "N": int(len(rows)), "rho": rho_o, "rho_sklearn_check": chk,
            "rho_gauss": rho_g, "rho_over_gauss": rho_o / rho_g if rho_g > 0 else None,
            "beta2_obs": b2o, "beta2_gauss": b2g, "delta2": d2,
            "beta2_obs_over_gauss": b2o / b2g if b2g > 0 else None,
            "lambda1_over_lambda2_raw": r_raw, "lambda1_over_lambda2_shrunk": r_shr,
            "PR_raw": PR, "excess_kurtosis_v1": kurt_v1,
            "excess_kurtosis_bulk50": kurt_bulk,
        }

    # full-sample spectrum shrunk by the TRAIN-selected rho (Julia's proposal)
    Xc_f = within_class_center(X, y)
    C_f = (Xc_f.T @ Xc_f) / len(y)
    mu_f = float(np.trace(C_f) / C_f.shape[0])
    _, _, r_cross, _ = spectrum(C_f, out["train"]["rho"], mu_f)
    out["full_shrunk_by_train_rho"] = {"lambda1_over_lambda2": r_cross}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma list; default all")
    args = ap.parse_args()
    want = {int(s) for s in args.layers.split(",")} if args.layers else None

    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        z = np.load(p); y = z["y"]
        layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
        if want:
            layers = [L for L in layers if L in want]
        print(f"\n=== {key} ===", flush=True)
        print(f"{'L':>4} {'rho_tr':>8} {'rho_full':>9} {'rho/gauss':>10} "
              f"{'kurt_v1':>9} {'kurt_bulk':>10} {'l1/l2 tr':>10} {'l1/l2 full':>11} "
              f"{'l1/l2 f@rho_tr':>15}", flush=True)
        per_layer = {}
        for L in layers:
            r = analyze(z[f"L{L}"].astype(np.float64), y, seed=args.seed)
            per_layer[str(L)] = r
            t, f = r["train"], r["full"]
            print(f"{L:>4} {t['rho']:>8.4f} {f['rho']:>9.4f} "
                  f"{t['rho_over_gauss']:>10.1f} {t['excess_kurtosis_v1']:>9.2f} "
                  f"{t['excess_kurtosis_bulk50']:>10.2f} "
                  f"{t['lambda1_over_lambda2_raw']:>10.1f} "
                  f"{f['lambda1_over_lambda2_raw']:>11.1f} "
                  f"{r['full_shrunk_by_train_rho']['lambda1_over_lambda2']:>15.1f}",
                  flush=True)
        results[key] = per_layer

    with open(args.out, "w") as fh:
        json.dump({"config": {"seed": args.seed,
                              "split": "class-stratified 50/50",
                              "estimator": "sklearn LedoitWolf(assume_centered=True)"},
                   "models": results}, fh, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
