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

This script measures a and b per dataset at a fixed model, averaging over
several label permutations per N, and records the spectrum diagnostics
alongside so the two can be compared.

The control is scored in-sample by design, so no half has to be held back and
the grid can run to the full set. Each dataset therefore gets its own geometric
grid from N=100 to its own N_total rather than one shared grid truncated at the
smallest dataset -- `sp_en_trans` (N=354) got two points under the shared grid,
which is a zero-residual fit and an uninformative exponent.

The spectrum is also recomputed at each N, not only on the full set, because
the collapse excess ~ C sqrt(PR/N) treats PR as a measured quantity: if the
sample PR drifts with the number of samples it was estimated from, C inherits
that drift and the small datasets are penalised for being small.

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
N_MIN = 100
N_PTS = 7           # geometric points from N_MIN to each dataset's own N_total
N_REP = 16
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


def grid_for(n_total):
    """Geometric N grid from N_MIN up to and including the full set."""
    if n_total <= N_MIN:
        return [int(n_total)]
    g = np.geomspace(N_MIN, n_total, N_PTS)
    return sorted({int(round(v)) for v in g} | {int(n_total)})


def excess_curve(X, y, d_model, ns, seed=0):
    """Mean in-sample shuffled-label excess AUROC at each N, over N_REP draws.

    The spectrum is recomputed on the first subsample at each N so that PR can
    be checked for drift with sample size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for N in ns:
        if N > len(y):
            continue
        exc, tru = [], []
        spec_at_N, first = None, True
        for _ in range(N_REP):
            idx = rng.choice(len(y), size=N, replace=False)
            Xs, ys = X[idx], y[idx]
            if len(np.unique(ys)) < 2:
                continue
            if first:
                spec_at_N = spectrum(Xs, ys)
                first = False
            ysh = rng.permutation(ys)
            th = S.mass_mean_direction(Xs, ysh)
            exc.append(S.evaluate_direction(th, Xs, ysh)["auroc"] - 0.5)
            tht = S.mass_mean_direction(Xs, ys)
            tru.append(S.evaluate_direction(tht, Xs, ys)["auroc"])
        rows.append({"N": int(N), "N_over_2d": float(N / (2 * d_model)),
                     "excess_mean": float(np.mean(exc)),
                     "excess_sd": float(np.std(exc, ddof=1)),
                     "true_auroc_mean": float(np.mean(tru)),
                     "n_rep": len(exc),
                     "exhaustive": bool(N == len(y)),
                     "pr_at_N": float(spec_at_N["participation_ratio"]),
                     "lambda1_over_trace_at_N": float(spec_at_N["lambda1_over_trace"])})
    return rows


def powerlaw_fit(rows):
    """OLS and inverse-variance-weighted LS of log(excess) on log(N/2d)."""
    x = np.log(np.array([r["N_over_2d"] for r in rows]))
    yv = np.log(np.array([max(r["excess_mean"], 1e-6) for r in rows]))
    b, loga = np.polyfit(x, yv, 1)
    resid = yv - (loga + b * x)
    ss = 1.0 - resid.var() / yv.var() if yv.var() > 0 else float("nan")
    out = {"amplitude": float(np.exp(loga)), "exponent": float(b),
           "r2": float(ss), "n_points": int(len(rows))}
    # log-space sd is sd/mean to first order; weight by its inverse square.
    rel = np.array([max(r["excess_sd"], 1e-9) / max(r["excess_mean"], 1e-9)
                    / max(np.sqrt(r["n_rep"]), 1.0) for r in rows])
    w = 1.0 / rel ** 2
    if len(rows) > 2 and np.all(np.isfinite(w)):
        bw, logaw = np.polyfit(x, yv, 1, w=np.sqrt(w))
        out["amplitude_wls"] = float(np.exp(logaw))
        out["exponent_wls"] = float(bw)
    return out


def main():
    tok, model = S.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    out = {"config": {"model": MODEL, "datasets": DATASETS,
                      "n_min": N_MIN, "n_pts": N_PTS, "cap": CAP,
                      "n_rep": N_REP,
                      "note": "in-sample shuffled-label excess; per-dataset "
                              "geometric N grid to the full set"},
           "results": {}}
    for ds in DATASETS:
        try:
            best = SWEEP["models"][MODEL]["datasets"][ds]["best_layer"]
            stmts, y = S.load_dataset(ds, cap=CAP, seed=0)
            print(f"[{ds}] N={len(y)} best_layer={best}", flush=True)
            A = S.extract_all_layers(stmts, tok, model, DEV, 16)
            X = A[best].astype(np.float64)
            d_model = X.shape[1]
            ns = grid_for(len(y))
            print(f"   grid {ns}", flush=True)
            rows = excess_curve(X, y, d_model, ns, seed=0)
            fit = powerlaw_fit(rows)
            out["results"][ds] = {"best_layer": int(best), "N_total": int(len(y)),
                                  "ns": ns,
                                  "spectrum": spectrum(X, y), "curve": rows,
                                  "fit": fit}
            print(f"   fit a={fit['amplitude']:.4f} b={fit['exponent']:+.3f} "
                  f"R2={fit['r2']:.3f}  PR={out['results'][ds]['spectrum']['participation_ratio']:.1f}",
                  flush=True)
            if "exponent_wls" in fit:
                print(f"   wls a={fit['amplitude_wls']:.4f} b={fit['exponent_wls']:+.3f}",
                      flush=True)
            for r in rows:
                print(f"     N={r['N']:>5} N/2d={r['N_over_2d']:.3f} "
                      f"excess={r['excess_mean']:.4f}±{r['excess_sd']:.4f} "
                      f"true={r['true_auroc_mean']:.3f} PR@N={r['pr_at_N']:.1f}",
                      flush=True)
            del A, X
            gc.collect()
        except Exception as e:
            print(f"  !! {ds} failed: {type(e).__name__}: {e}", flush=True)
            out["results"][ds] = {"error": f"{type(e).__name__}: {e}"}
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  -> wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
