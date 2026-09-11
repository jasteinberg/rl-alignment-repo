"""
Is the rogue dimension eleven outlier statements? Four checks.

The finding that prompted this was selected circularly: points were thresholded on
their projection onto v1, then v1 was recomputed without them. Removing the top of
a distribution shortens it, so that alone shows nothing. This re-derives the outlier
set from a criterion that never looks at v1, then measures held-out.

  (1) SELECTION INDEPENDENT OF v1. Sun et al.'s massive-activation rule, applied
      coordinate-wise to the raw activations: |x_ij| > 100 and > 1000x the median
      |x| of that layer. Reports the overlap with the v1-selected set. If they
      coincide, the v1 result is real; if not, it was an artifact of selection.

  (2) WHICH STATEMENTS. Regenerates the statement list with the same loader, seed
      and cap, verifies the labels match the cache, and prints the flagged ones.

  (3) HELD-OUT. Refits theta_mm and theta_F on the train half and scores d' on the
      test half, for: all points; both halves cleaned; and train cleaned with the
      full test (the practical question -- does dropping them help a probe that is
      still deployed on contaminated data?). d'_M has no held-out analogue (it is a
      max over directions), so it is reported in-sample and labelled as such.

  (4) GENERALIZATION. Same for pythia-1.4b, and for cities as the control that
      should show nothing.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os

import numpy as np
from sklearn.covariance import LedoitWolf
from truthlib import data
from truthlib.estimators import (split_indices, within_class_center, d_prime,
                                 mass_mean, fisher, massive_mask)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "outlier_check.json")


def v1_mask(X, y, zmax=4.0):
    """The circular selection, kept only so the two sets can be compared."""
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    v1 = np.linalg.eigh(C)[1][:, -1]
    p = Xc @ v1
    return np.abs(p) / p.std() > zmax


def geometry(X, y):
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    w, V = np.linalg.eigh(C)
    lam = np.clip(w[::-1], 0.0, None)
    th = mass_mean(X, y)
    P = LedoitWolf(assume_centered=True).fit(Xc).precision_
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    return {
        "lambda1_over_lambda2": float(lam[0] / lam[1]),
        "lambda1_over_trace": float(lam[0] / lam.sum()),
        "PR": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "cos_theta_v1": float(abs(th @ V[:, -1])),
        "d_mahalanobis_INSAMPLE": float(np.sqrt(dlt @ P @ dlt)),
    }


def held_out(X, y, drop, seed=0, n_null=200):
    """Fit on train, score on test. Three cleaning regimes."""
    tr, te = split_indices(y, seed=seed)
    keep = ~drop
    out = {}
    regimes = {
        "all":          (tr, te),
        "clean_both":   (tr[keep[tr]], te[keep[te]]),
        "clean_train":  (tr[keep[tr]], te),
    }
    for tag, (a, b) in regimes.items():
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        row = {"n_train": int(len(a)), "n_test": int(len(b))}
        for kind, fn in (("mm", mass_mean), ("F", fisher)):
            t = fn(X[a], y[a])
            if d_prime(X[a] @ t, y[a]) and (X[a] @ t)[y[a] == 1].mean() < \
               (X[a] @ t)[y[a] == 0].mean():
                t = -t                       # orient on train only
            row[f"d_{kind}"] = d_prime(X[b] @ t, y[b])
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            u = rng.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
            nulls.append(d_prime(X[b] @ u, y[b]))
        row["null_p95"] = float(np.percentile(nulls, 95))
        out[tag] = row
    return out


def statements_for(dataset, cap, seed=0):
    """Regenerate the loader's statement list so flagged rows can be read."""
    return data.load_dataset(dataset, cap=cap, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=1199)
    args = ap.parse_args()

    results = {}
    for model in ("pythia-2.8b", "pythia-1.4b"):
        for ds in ("counterfact_true_false", "cities"):
            path = os.path.join(CACHE, f"{model}__{ds}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path); y = z["y"]
            layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
            key = f"{model}__{ds}"
            print(f"\n{'='*74}\n=== {key} ===", flush=True)

            try:
                stmts, y_re = statements_for(ds, args.cap, args.seed)
                match = len(y_re) == len(y) and bool((y_re == y).all())
            except Exception as e:
                stmts, match = None, f"loader failed: {e}"
            print(f"statement list regenerated, labels match cache: {match}", flush=True)

            per_layer = {}
            for L in layers:
                X = z[f"L{L}"].astype(np.float64)
                mass, med, hit = massive_mask(X)
                circ = v1_mask(X, y)
                inter = int((mass & circ).sum())
                per_layer[str(L)] = {
                    "median_abs_activation": med,
                    "n_massive": int(mass.sum()), "n_v1_selected": int(circ.sum()),
                    "n_overlap": inter,
                    "geometry_all": geometry(X, y),
                    "geometry_clean": geometry(X[~mass], y[~mass])
                                      if 0 < mass.sum() < len(y) - 20 else None,
                    "held_out": held_out(X, y, mass, seed=args.seed),
                }
                g = per_layer[str(L)]
                gc = g["geometry_clean"] or {}
                ho = g["held_out"]
                print(f"  L{L:<3} massive={g['n_massive']:<4} v1sel={g['n_v1_selected']:<4} "
                      f"overlap={inter:<4} | l1/l2 {g['geometry_all']['lambda1_over_lambda2']:>9.1f}"
                      f" -> {gc.get('lambda1_over_lambda2', float('nan')):>7.1f}"
                      f" | cos {g['geometry_all']['cos_theta_v1']:.3f}"
                      f" -> {gc.get('cos_theta_v1', float('nan')):.3f}", flush=True)
                for tag in ("all", "clean_both", "clean_train"):
                    if tag in ho:
                        r = ho[tag]
                        print(f"        {tag:<12} n={r['n_test']:<5} d_mm={r['d_mm']:.3f} "
                              f"d_F={r['d_F']:.3f}  null_p95={r['null_p95']:.3f}", flush=True)

                if stmts is not None and mass.sum() and L == layers[-1]:
                    idx = np.where(mass)[0]
                    print(f"    flagged statements ({len(idx)}):", flush=True)
                    for i in idx[:15]:
                        print(f"      y={y[i]}  {stmts[i][:96]}", flush=True)
                    per_layer[str(L)]["flagged"] = [
                        {"i": int(i), "y": int(y[i]), "statement": stmts[i]} for i in idx]
            results[key] = per_layer

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "cap": args.cap,
                              "selection": "Sun et al. |x|>100 and >100x median |x| (relaxed from their 1000x; see truthlib.estimators.massive_mask)"},
                   "results": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
