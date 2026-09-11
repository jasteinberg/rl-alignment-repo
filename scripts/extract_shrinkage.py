"""
Extract the Ledoit-Wolf shrinkage intensity rho actually used by the whitened
(Fisher) direction, per layer, from the cached activations.

snr_sweep.py fits LedoitWolf(assume_centered=True) on the TRAIN half of the
within-class-centered activations and never records lw.shrinkage_, so the post's
methods appendix names the estimator without being able to quote the number.
This reproduces that fit exactly -- same class-stratified 50/50 split at seed 0,
same within-class centering, same float64 cast -- and reports rho along with the
raw and shrunk spectra so the effective anisotropy can be compared against the
raw lambda_1/lambda_2 the plane-rotation section uses.

Reports per layer:
  rho          Ledoit-Wolf shrinkage intensity (0 = raw C, 1 = isotropic)
  l1/l2 raw    leading eigenvalue ratio of the raw within-class C (train half)
  l1/l2 shr    the same ratio after shrinkage, (1-rho)*l + rho*mu
  l1/tr raw    leading eigenvalue share of the raw C
  PR raw       participation ratio of the raw C

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf
from truthlib.estimators import split_indices, within_class_center

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "shrinkage_intensity.json")


def layer_row(X, y, seed=0):
    tr, _ = split_indices(y, seed=seed)
    Xtr, ytr = X[tr], y[tr]
    Xc = within_class_center(Xtr, ytr)

    lw = LedoitWolf(assume_centered=True).fit(Xc)
    rho = float(lw.shrinkage_)

    # Raw within-class covariance on the same rows (1/N convention, as LW uses).
    C = (Xc.T @ Xc) / Xc.shape[0]
    lam = np.linalg.eigvalsh(C)[::-1]
    lam = np.clip(lam, 0.0, None)
    mu = float(np.trace(C) / C.shape[0])
    lam_shr = (1.0 - rho) * lam + rho * mu

    def ratio(v):
        return float(v[0] / v[1]) if v[1] > 0 else float("inf")

    return {
        "n_train": int(Xtr.shape[0]),
        "d": int(X.shape[1]),
        "rho": rho,
        "lambda1_over_lambda2_raw": ratio(lam),
        "lambda1_over_lambda2_shrunk": ratio(lam_shr),
        "lambda1_over_trace_raw": float(lam[0] / lam.sum()),
        "PR_raw": float(lam.sum() ** 2 / (lam ** 2).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        z = np.load(p)
        y = z["y"]
        layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
        print(f"\n=== {key} ===", flush=True)
        print(f"{'L':>4} {'rho':>8} {'l1/l2 raw':>12} {'l1/l2 shr':>12} "
              f"{'l1/tr raw':>10} {'PR raw':>8}", flush=True)
        per_layer = {}
        for L in layers:
            r = layer_row(z[f"L{L}"].astype(np.float64), y, seed=args.seed)
            per_layer[str(L)] = r
            print(f"{L:>4} {r['rho']:>8.4f} {r['lambda1_over_lambda2_raw']:>12.1f} "
                  f"{r['lambda1_over_lambda2_shrunk']:>12.1f} "
                  f"{r['lambda1_over_trace_raw']:>10.3f} {r['PR_raw']:>8.2f}",
                  flush=True)
        results[key] = per_layer

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed,
                              "split": "class-stratified 50/50, train half",
                              "estimator": "sklearn LedoitWolf(assume_centered=True)"},
                   "models": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
