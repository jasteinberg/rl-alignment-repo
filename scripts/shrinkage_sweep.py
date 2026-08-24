"""
Does shrinkage explain why whitening falls short below layer 24?

The post's new section observes that on counterfact at pythia-2.8b, removing the eleven
massive-activation droppers raises held-out d'_F at L8-L20 (0.005-0.031 -> 0.129-0.385)
while leaving L28 untouched (0.384 -> 0.385). The hypothesis in the text is that
Ledoit-Wolf's rho ~ 0.137 is too large to let

    Sigma_hat(rho) = (1 - rho) * C_hat + rho * mu * I,     mu = tr C_hat / d

invert an anisotropy of order 1e4, so theta_F = Sigma_hat(rho)^-1 delta_hat cannot rotate
far enough off v1.

PREDICTION IF TRUE: sweeping rho DOWN on the uncleaned data should recover some of the
gain that deleting the eleven produces -- d'_F at L8-L20 should rise as rho falls, toward
the cleaned values. PREDICTION IF FALSE: d'_F stays flat or degrades at every rho, and the
shortfall is about something other than regularization strength.

rho -> 0 is not available: C_hat is singular at N_train ~ 599 against d = 2560, so the
grid stops at 1e-4 and the inverse is taken via solve on the shrunk matrix, which stays
positive definite for any rho > 0.

Everything is fit on the train half and scored on the full held-out half, matching the
"clean_train" regime of outlier_check.py so the numbers are directly comparable.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "shrinkage_sweep.json")
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 0.9]


def split_indices(y, frac=0.5, seed=0):
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


def d_prime(z, y):
    a, b = z[y == 1], z[y == 0]
    return float(abs(a.mean() - b.mean()) /
                 np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))


def fisher_at_rho(Xtr, ytr, rho):
    Xc = within_class_center(Xtr, ytr)
    C = (Xc.T @ Xc) / len(ytr)
    mu = np.trace(C) / C.shape[0]
    S = (1.0 - rho) * C + rho * mu * np.eye(C.shape[0])
    dlt = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    t = np.linalg.solve(S, dlt)
    return t / np.linalg.norm(t)


def droppers(X, mag=100.0, rel=100.0, frac=0.5):
    """Same coordinate-first rule as outlier_check.py."""
    med_all = float(np.median(np.abs(X)))
    med_j = np.median(X, axis=0)
    massive = (np.abs(med_j) > mag) & (np.abs(med_j) > rel * med_all)
    if not massive.any():
        return np.zeros(len(X), bool)
    return (np.abs(X[:, massive] - med_j[massive]) >
            frac * np.abs(med_j[massive])).any(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    args = ap.parse_args()

    z = np.load(os.path.join(CACHE, f"{args.model}__{args.dataset}.npz"))
    y = z["y"]
    tr, te = split_indices(y, seed=args.seed)
    layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))

    results = {}
    hdr = "  ".join(f"{r:>7.4g}" for r in GRID)
    print(f"{'L':>4} {'rho_LW':>7} {'d_LW':>6} {'d_clean':>8} | {hdr}", flush=True)
    for L in layers:
        X = z[f"L{L}"].astype(np.float64)
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        rho_lw = float(LedoitWolf(assume_centered=True)
                       .fit(within_class_center(Xtr, ytr)).shrinkage_)

        def score(t):
            if (Xtr @ t)[ytr == 1].mean() < (Xtr @ t)[ytr == 0].mean():
                t = -t
            return d_prime(Xte @ t, yte)

        d_lw = score(fisher_at_rho(Xtr, ytr, rho_lw))

        # reference: the eleven removed from train only, at LW's own rho there
        keep = ~droppers(Xtr)
        Xc2, yc2 = Xtr[keep], ytr[keep]
        rho_c = float(LedoitWolf(assume_centered=True)
                      .fit(within_class_center(Xc2, yc2)).shrinkage_)
        tc = fisher_at_rho(Xc2, yc2, rho_c)
        if (Xc2 @ tc)[yc2 == 1].mean() < (Xc2 @ tc)[yc2 == 0].mean():
            tc = -tc
        d_clean = d_prime(Xte @ tc, yte)

        row = {"rho_LW": rho_lw, "d_F_at_rho_LW": d_lw,
               "rho_LW_clean": rho_c, "d_F_clean_train": d_clean, "sweep": {}}
        vals = []
        for rho in GRID:
            v = score(fisher_at_rho(Xtr, ytr, rho))
            row["sweep"][str(rho)] = v
            vals.append(v)
        results[str(L)] = row
        print(f"{L:>4} {rho_lw:>7.4f} {d_lw:>6.3f} {d_clean:>8.3f} | "
              + "  ".join(f"{v:>7.3f}" for v in vals), flush=True)

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "model": args.model,
                              "dataset": args.dataset, "grid": GRID,
                              "note": "fit on train half, scored on FULL held-out half"},
                   "layers": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
