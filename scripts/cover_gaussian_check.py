"""
Why does sp_en_trans miss the sqrt(PR/N) collapse?

The collapse constant is not free. Under shuffled labels there is no signal, so
x ~ N(0, Sigma) within class and the mass-mean direction is

    theta = mu_+ - mu_- = (2/N) sum_i eps_i x_i,   eps_i = +-1,

so theta ~ N(0, (4/N) Sigma). The in-sample separation it produces is

    E|theta|^2       = (4/N) tr Sigma
    E theta' Sigma theta = (4/N) tr Sigma^2

    d'_in = |theta|^2 / sqrt(theta' Sigma theta) = 2 sqrt(PR / N),

with PR = (tr Sigma)^2 / tr Sigma^2. If the projections are Gaussian then
AUROC = Phi(d'/sqrt2), and for small d'

    AUROC - 1/2 ~ d' / (2 sqrt(pi)) = sqrt(PR/N) / sqrt(pi),

i.e. C = 1/sqrt(pi) = 0.5642, a parameter-free prediction.

That splits the collapse into two independent steps:

  (A) geometry     d'_in  =?  2 sqrt(PR/N)      -- second moments only
  (B) Gaussianity  AUROC  =?  Phi(d'_in/sqrt2)  -- shape of the projection

C can be inflated by a failure of either. This script measures both separately
per dataset, plus the excess kurtosis of the within-class projection, so the
sp_en_trans anomaly can be localised to one step.

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import json
import math
import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, kurtosis

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("s", REPO / "scripts" / "snr_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

MODEL = os.environ.get("COVER_MODEL", "EleutherAI/pythia-2.8b")
DATASETS = ["counterfact_true_false", "cities", "larger_than", "sp_en_trans"]
NS = [100, 200, 354]
N_REP = 32
CAP = 3000
DEV = os.environ.get("COVER_DEV", "mps")
OUT = REPO / "artifacts" / "cover_gaussian_check.json"
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))


def pr_of(X, y):
    Xc = np.vstack([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    lam = np.clip(np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1], 0.0, None)
    tr = lam.sum()
    return float(tr ** 2 / (lam ** 2).sum()), lam


def one_draw(X, y, rng, N):
    """One shuffled-label fit: return measured AUROC, d', and projection shape."""
    idx = rng.choice(len(y), size=N, replace=False)
    Xs = X[idx]
    ysh = rng.permutation(y[idx])
    if len(np.unique(ysh)) < 2:
        return None
    th = S.mass_mean_direction(Xs, ysh)
    z = Xs @ th
    z1, z0 = z[ysh == 1], z[ysh == 0]
    # pooled within-class sd of the projection
    sp = math.sqrt(0.5 * (z1.var(ddof=1) + z0.var(ddof=1)))
    dprime = float((z1.mean() - z0.mean()) / sp) if sp > 0 else float("nan")
    auroc = float(S.evaluate_direction(th, Xs, ysh)["auroc"])
    zc = np.concatenate([z1 - z1.mean(), z0 - z0.mean()])
    return dprime, auroc, float(kurtosis(zc, fisher=True))


def main():
    tok, model = S.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    out = {"config": {"model": MODEL, "ns": NS, "n_rep": N_REP,
                      "C_predicted": 1.0 / math.sqrt(math.pi)},
           "results": {}}
    for ds in DATASETS:
        best = SWEEP["models"][MODEL]["datasets"][ds]["best_layer"]
        stmts, y = S.load_dataset(ds, cap=CAP, seed=0)
        A = S.extract_all_layers(stmts, tok, model, DEV, 16)
        X = A[best].astype(np.float64)
        PR, lam = pr_of(X, y)
        print(f"[{ds}] N_total={len(y)} L={best} PR={PR:.1f}", flush=True)
        rows = []
        rng = np.random.default_rng(0)
        for N in NS:
            if N > len(y):
                continue
            dp, au, ku = [], [], []
            for _ in range(N_REP):
                r = one_draw(X, y, rng, N)
                if r:
                    dp.append(r[0]); au.append(r[1]); ku.append(r[2])
            dp_m, au_m, ku_m = float(np.mean(dp)), float(np.mean(au)), float(np.mean(ku))
            dp_pred = 2.0 * math.sqrt(PR / N)
            au_from_dp = float(norm.cdf(dp_m / math.sqrt(2)))
            rows.append({
                "N": N, "dprime_meas": dp_m, "dprime_pred": dp_pred,
                "step_A_ratio": dp_m / dp_pred,
                "auroc_meas": au_m, "auroc_from_dprime": au_from_dp,
                "step_B_ratio": (au_m - 0.5) / (au_from_dp - 0.5),
                "kurtosis": ku_m,
                "C_meas": (au_m - 0.5) * math.sqrt(N / PR),
            })
            r = rows[-1]
            print(f"   N={N:5d}  d'meas={dp_m:.3f} d'pred={dp_pred:.3f} (A={r['step_A_ratio']:.3f})"
                  f"  AUROC={au_m:.4f} from-d'={au_from_dp:.4f} (B={r['step_B_ratio']:.3f})"
                  f"  kurt={ku_m:+.2f}  C={r['C_meas']:.3f}", flush=True)
        # spectral shape beyond PR: normalised eigenvalue moments
        p = lam / lam.sum()
        out["results"][ds] = {
            "best_layer": int(best), "N_total": int(len(y)), "PR": PR,
            "participation_ratios": {
                "PR2": float(1.0 / (p ** 2).sum()),
                "PR3": float(1.0 / (p ** 3).sum() ** 0.5),
                "shannon_exp": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
                "lambda1_over_trace": float(p[0]),
                "top10_share": float(p[:10].sum()),
                "top50_share": float(p[:50].sum()),
            },
            "curve": rows,
        }
        del A, X
        gc.collect()
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    main()
