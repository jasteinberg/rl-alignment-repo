"""
Does the shuffled-label amplitude depend on the dataset?

The blog fits AUROC_shuffled - 1/2 ~ a (N/2d)^b on `counterfact` only, pooling
four models, and gets a = 0.045, b = -0.49. The exponent is predicted by the
estimator: a difference of two sample means of noise is O(sqrt(d/N)), so the
in-sample excess must fall as N^{-1/2} at fixed d whatever the covariance. The
amplitude has no such derivation -- it should absorb the SHAPE of the
within-class spectrum, i.e. how many directions the noise effectively occupies.

Effective-dimension hypothesis: if the spurious separation scales as
sqrt(d_eff/N) with d_eff the participation ratio (sum lam)^2 / sum lam^2 of the
within-class covariance, then a dataset with a strongly concentrated spectrum
(small PR) should sit LOWER, not higher.

This script measures a and b per dataset at a fixed model, on a shared N grid,
averaging over several label permutations per N, and records the spectrum
diagnostics alongside so the two can be compared.

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import json
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("s", REPO / "scripts" / "snr_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

MODEL = os.environ.get("COVER_MODEL", "EleutherAI/pythia-2.8b")
DATASETS = ["counterfact_true_false", "cities", "larger_than", "sp_en_trans"]
NS = [100, 200, 400, 800, 1200]
N_REP = 8
CAP = 3000          # applies only to counterfact (N=31,964); the rest are smaller
DEV = os.environ.get("COVER_DEV", "mps")
OUT = REPO / "artifacts" / "cover_by_dataset.json"
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))


def spectrum(X, y):
    """Within-class covariance diagnostics at this layer."""
    Xc = np.vstack([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    lam = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1]
    lam = np.clip(lam, 0.0, None)
    tr = lam.sum()
    return {
        "participation_ratio": float(tr ** 2 / (lam ** 2).sum()),
        "lambda1_over_trace": float(lam[0] / tr),
        "kappa_12": float(lam[0] / max(lam[1], 1e-12)),
        "d_model": int(X.shape[1]),
    }


def excess_curve(X, y, d_model, seed=0):
    """Mean in-sample shuffled-label excess AUROC at each N, over N_REP draws."""
    rng = np.random.default_rng(seed)
    rows = []
    for N in NS:
        if N > len(y):
            continue
        exc, tru = [], []
        for _ in range(N_REP):
            idx = rng.choice(len(y), size=N, replace=False)
            Xs, ys = X[idx], y[idx]
            if len(np.unique(ys)) < 2:
                continue
            ysh = rng.permutation(ys)
            th = S.mass_mean_direction(Xs, ysh)
            exc.append(S.evaluate_direction(th, Xs, ysh)["auroc"] - 0.5)
            tht = S.mass_mean_direction(Xs, ys)
            tru.append(S.evaluate_direction(tht, Xs, ys)["auroc"])
        rows.append({"N": int(N), "N_over_2d": float(N / (2 * d_model)),
                     "excess_mean": float(np.mean(exc)),
                     "excess_sd": float(np.std(exc, ddof=1)),
                     "true_auroc_mean": float(np.mean(tru)),
                     "n_rep": len(exc)})
    return rows


def powerlaw_fit(rows):
    """OLS of log(excess) on log(N/2d): excess ~ a (N/2d)^b."""
    x = np.log(np.array([r["N_over_2d"] for r in rows]))
    yv = np.log(np.array([max(r["excess_mean"], 1e-6) for r in rows]))
    b, loga = np.polyfit(x, yv, 1)
    resid = yv - (loga + b * x)
    ss = 1.0 - resid.var() / yv.var() if yv.var() > 0 else float("nan")
    return {"amplitude": float(np.exp(loga)), "exponent": float(b),
            "r2": float(ss), "n_points": int(len(rows))}


def main():
    tok, model = S.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    out = {"config": {"model": MODEL, "datasets": DATASETS, "ns": NS,
                      "n_rep": N_REP, "note": "in-sample shuffled-label excess"},
           "results": {}}
    for ds in DATASETS:
        try:
            best = SWEEP["models"][MODEL]["datasets"][ds]["best_layer"]
            stmts, y = S.load_dataset(ds, cap=CAP, seed=0)
            print(f"[{ds}] N={len(y)} best_layer={best}", flush=True)
            A = S.extract_all_layers(stmts, tok, model, DEV, 16)
            X = A[best].astype(np.float64)
            d_model = X.shape[1]
            rows = excess_curve(X, y, d_model, seed=0)
            fit = powerlaw_fit(rows)
            out["results"][ds] = {"best_layer": int(best), "N_total": int(len(y)),
                                  "spectrum": spectrum(X, y), "curve": rows,
                                  "fit": fit}
            print(f"   fit a={fit['amplitude']:.4f} b={fit['exponent']:+.3f} "
                  f"R2={fit['r2']:.3f}  PR={out['results'][ds]['spectrum']['participation_ratio']:.1f}",
                  flush=True)
            for r in rows:
                print(f"     N={r['N']:>5} N/2d={r['N_over_2d']:.3f} "
                      f"excess={r['excess_mean']:.4f}±{r['excess_sd']:.4f} "
                      f"true={r['true_auroc_mean']:.3f}", flush=True)
            del A, X
            gc.collect()
        except Exception as e:
            print(f"  !! {ds} failed: {type(e).__name__}: {e}", flush=True)
            out["results"][ds] = {"error": f"{type(e).__name__}: {e}"}
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  -> wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
