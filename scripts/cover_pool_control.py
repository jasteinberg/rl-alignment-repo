"""
Is the sp_en_trans anomaly a property of the dataset, or of measuring PR from
only 354 samples?

cover_gaussian_check.py localised the miss to step (A): sp_en_trans's in-sample
d' runs ~50% above 2 sqrt(PR/N) where the other three sit within 3-17%. Step
(B), AUROC = Phi(d'/sqrt2), holds to 2% everywhere, so the projection shape is
not the culprit.

PR here is a SAMPLE participation ratio. Sigma-hat from P samples in d=2560 has
rank <= P-2, and PR@N was already seen to rise with N and not saturate until
N ~ 1000 on counterfact. sp_en_trans has N_total = 354, so its PR can only ever
be read at the small-P end of that curve. If that is the whole story, then
restricting a LARGE dataset to a pool of 354 and re-measuring PR from that pool
should reproduce the same inflated A.

That is the control: vary the pool size P at fixed dataset and watch A. If A(P)
falls toward 1 as P grows, the anomaly belongs to the estimate of PR, not to
sp_en_trans. If counterfact holds A ~ 1 even at P = 354, sp_en_trans is
genuinely different.

Also decomposes A into its numerator and denominator against the second-moment
prediction:

    num = |theta|^2  / ((4/N) tr Sigma)
    den = theta' Sigma theta / ((4/N) tr Sigma^2)
    A   = d'_meas / (2 sqrt(PR/N)) = num / sqrt(den)

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import json
import math
import os
from truthlib import acts, data
from truthlib import estimators as est
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

MODEL = os.environ.get("COVER_MODEL", "EleutherAI/pythia-2.8b")
DATASETS = ["counterfact_true_false", "cities", "larger_than", "sp_en_trans"]
POOLS = [354, 800, 1980, 3000]
NS = [100, 200, 354]
N_REP = 24
N_POOL_REP = 4
CAP = 3000
DEV = os.environ.get("COVER_DEV", "mps")
OUT = REPO / "artifacts" / "cover_pool_control.json"
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))


def within_cov(X, y):
    Xc = np.vstack([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    return np.cov(Xc, rowvar=False)


def pool_stats(X, y):
    Sig = within_cov(X, y)
    lam = np.clip(np.linalg.eigvalsh(Sig)[::-1], 0.0, None)
    tr, tr2 = lam.sum(), (lam ** 2).sum()
    return Sig, float(tr), float(tr2), float(tr ** 2 / tr2)


def main():
    tok, model = acts.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    out = {"config": {"model": MODEL, "pools": POOLS, "ns": NS,
                      "n_rep": N_REP, "n_pool_rep": N_POOL_REP}, "results": {}}
    for ds in DATASETS:
        best = SWEEP["models"][MODEL]["datasets"][ds]["best_layer"]
        stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
        A_all = acts.extract_all_layers(stmts, tok, model, DEV, 16)
        X = A_all[best].astype(np.float64)
        print(f"[{ds}] N_total={len(y)} L={best}", flush=True)
        rows = []
        rng = np.random.default_rng(0)
        for P in POOLS:
            if P > len(y):
                continue
            for _ in range(N_POOL_REP if P < len(y) else 1):
                pid = rng.choice(len(y), size=P, replace=False)
                Xp, yp = X[pid], y[pid]
                Sig, tr, tr2, PR = pool_stats(Xp, yp)
                for N in NS:
                    if N > P:
                        continue
                    a_s, num_s, den_s = [], [], []
                    for _ in range(N_REP):
                        idx = rng.choice(P, size=N, replace=False)
                        Xs = Xp[idx]
                        ysh = rng.permutation(yp[idx])
                        if len(np.unique(ysh)) < 2:
                            continue
                        th = est.mass_mean(Xs, ysh)
                        th = th * np.linalg.norm(th) / max(np.linalg.norm(th), 1e-12)
                        # mass_mean_direction returns a unit vector; rebuild the raw gap
                        mu1 = Xs[ysh == 1].mean(0)
                        mu0 = Xs[ysh == 0].mean(0)
                        raw = mu1 - mu0
                        n2 = float(raw @ raw)
                        q = float(raw @ (Sig @ raw))
                        z = Xs @ raw
                        z1, z0 = z[ysh == 1], z[ysh == 0]
                        sp = math.sqrt(0.5 * (z1.var(ddof=1) + z0.var(ddof=1)))
                        if sp <= 0:
                            continue
                        a_s.append(float((z1.mean() - z0.mean()) / sp)
                                   / (2.0 * math.sqrt(PR / N)))
                        num_s.append(n2 / ((4.0 / N) * tr))
                        den_s.append(q / ((4.0 / N) * tr2))
                    rows.append({"P": P, "PR_pool": PR, "N": N,
                                 "A": float(np.mean(a_s)),
                                 "num": float(np.mean(num_s)),
                                 "den": float(np.mean(den_s))})
        # average over pool repeats
        agg = {}
        for r in rows:
            agg.setdefault((r["P"], r["N"]), []).append(r)
        curve = []
        for (P, N), rs in sorted(agg.items()):
            curve.append({"P": P, "N": N,
                          "PR_pool": float(np.mean([r["PR_pool"] for r in rs])),
                          "A": float(np.mean([r["A"] for r in rs])),
                          "num": float(np.mean([r["num"] for r in rs])),
                          "den": float(np.mean([r["den"] for r in rs]))})
            c = curve[-1]
            print(f"   P={P:5d} N={N:4d}  PR_pool={c['PR_pool']:6.1f}  A={c['A']:.3f}"
                  f"   num={c['num']:.3f} den={c['den']:.3f}", flush=True)
        out["results"][ds] = {"best_layer": int(best), "N_total": int(len(y)),
                              "curve": curve}
        del A_all, X
        gc.collect()
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    main()
