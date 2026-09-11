"""
scripts/steer_confirm2.py

Does causal steering efficacy anticorrelate with decodability across depth, and
does whitening trade intervention efficacy for readout quality?

DESIGN

Decomposition. A steering effect must be split into
    antisym  A = [shift(+a) - shift(-a)] / 2    <- real steering
    sym      S = [shift(+a) + shift(-a)] / 2    <- perturbation damage
Only A is evidence of a causal direction: a generic out-of-distribution
perturbation degrades the forward pass for BOTH signs, contributing to S while
leaving A near zero. The pilot showed |z| up to 38 at mid layers with A ~ 0.

Seed allocation. Layers are fractions of depth, so models of different depth are
comparable. The early/mid window is where a causal variable is PREDICTED to live
(it must be upstream of the computation consuming it); the deepest layer is the
control, where decoding is near-perfect and the prediction is A ~ 0. Both ends of
the claim get 10 seeds, the transition region 5. The allocation follows the
prediction, not the pilot's observed peak -- otherwise the tightest error bars
would sit exactly where noise happened to be favourable.

Two noise sources. The split seed varies which statements fit theta (estimation
noise in the direction); the pair sample varies the readout. Both are resampled
per seed, so error bars cover resamples of the probing set AND the evaluation set.

Random controls involve no fitting, so they do not depend on the split seed: they
are drawn once per (model, dataset, layer, alpha) with n_rand draws.

RESUMABILITY

Every (model, dataset, layer, seed) cell is written to its own file under
artifacts/steer_ckpt/. Seed semantics are stable -- seed k always means the same
split and the same pair subsample -- so seeds can be appended later with
    --seeds 10,11,...,19
and nothing already computed is recomputed. Activations for the probed layers are
cached to artifacts/act_cache/, so a seed extension costs only its scoring passes.
The notebook loads whatever cells exist via load_cells().

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, gc, argparse
import numpy as np, torch
from truthlib import acts, data, steering
from truthlib import estimators as est
from truthlib.steering import (SEED_PLAN, cell_path, null_path, get_acts, antisym,
                               fit_dirs, pairs_for_seed)
from truthlib.estimators import class_gap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ART = os.path.join(REPO, "artifacts")
CKPT = os.path.join(ART, "steer_ckpt")  # class-gap alpha units
ACTS = os.path.join(ART, "act_cache")
DEV = "mps"

def run_cell(model, tok, Xl, y, pool, layer, alphas, seed, bs, dtype, n_pairs):
    X = Xl.astype(np.float64)
    th, thw, tr, te = fit_dirs(X, y, seed)
    pairs = pairs_for_seed(pool, seed, n_pairs)
    scale = class_gap(X[tr], y[tr])          # fit the unit on train, like theta
    rec = {"layer": layer, "seed": seed, "alpha_unit": scale,
           "alpha_unit_kind": "class_gap_norm",
           "sigma_along_theta": float(np.std(X @ th)),
           "d_prime_theta": scale / float(np.std(X @ th)),
           "n_pairs": len(pairs),
           "probe_auroc": est.auroc(X[te] @ th, y[te]),
           "probe_auroc_whitened": est.auroc(X[te] @ thw, y[te]), "alphas": {}}
    with steering.Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = steering.score_pairs(model, tok, pairs, bs)
        for a in alphas:
            ap, sp_ = antisym(model, tok, st, th, a, scale, pairs, base, bs, dtype)
            aw, sw_ = antisym(model, tok, st, thw, a, scale, pairs, base, bs, dtype)
            rec["alphas"][str(a)] = {"plain": {"antisym": ap, "sym": sp_},
                                     "whitened": {"antisym": aw, "sym": sw_}}
        st.set(None, 0, DEV, dtype)
    rec["baseline"] = float(base.mean())
    return rec, scale


def run_null(model, tok, Xl, y, pool, layer, alphas, bs, dtype, n_pairs, n_rand):
    X = Xl.astype(np.float64)
    scale = class_gap(X, y)                  # same unit as the theta arms
    pairs = pairs_for_seed(pool, 0, n_pairs)
    rng = np.random.default_rng(9000 + layer * 17)
    out = {"layer": layer, "n_rand": n_rand, "alphas": {}}
    with steering.Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = steering.score_pairs(model, tok, pairs, bs)
        for a in alphas:
            vals = []
            for _ in range(n_rand):
                r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
                av, _ = antisym(model, tok, st, r, a, scale, pairs, base, bs, dtype)
                vals.append(av)
            v = np.array(vals)
            out["alphas"][str(a)] = {"antisym_mean": float(v.mean()),
                                     "antisym_std": float(v.std(ddof=1)),
                                     "antisym_p95": float(np.percentile(np.abs(v), 95)),
                                     "n_draws": len(v),
                                     "draws": [float(x) for x in v]}
        st.set(None, 0, DEV, dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b,EleutherAI/pythia-1.4b")
    ap.add_argument("--datasets", default="cities,counterfact_true_false")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="", help="explicit seed list, e.g. 10,11,12; "
                    "default uses SEED_PLAN per layer")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_rand", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    explicit = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    os.makedirs(CKPT, exist_ok=True)

    for mname in [m for m in args.models.split(",") if m]:
        tok, model = acts.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        nL = model.config.num_hidden_layers
        plan = {max(1, int(round(f * nL))): n for f, n in SEED_PLAN.items()}
        print(f"\n=== {mname} ({nL} layers) plan={plan} ===", flush=True)

        for ds in [d for d in args.datasets.split(",") if d]:
            print(f"  [{ds}]", flush=True)
            layers = sorted(plan)
            Xs, y = get_acts(model, tok, mname, ds, layers, args.cap)
            pool = data.load_pairs(ds, args.pairs, 0)

            for L in layers:
                seeds = explicit if explicit is not None else list(range(plan[L]))
                for sd in seeds:
                    cp = cell_path(mname, ds, L, sd)
                    if os.path.exists(cp) and not args.force:
                        continue
                    rec, _ = run_cell(model, tok, Xs[L], y, pool, L, alphas,
                                      sd, args.bs, dtype, args.pairs_per_seed)
                    with open(cp, "w") as f:
                        json.dump(rec, f, indent=2)
                    a8 = rec["alphas"][str(alphas[-1])]
                    print(f"    L{L:<3} s{sd:<3} AUROC={rec['probe_auroc']:.3f} "
                          f"plain A={a8['plain']['antisym']:+.3f} "
                          f"whit A={a8['whitened']['antisym']:+.3f}", flush=True)

                np_ = null_path(mname, ds, L)
                if not os.path.exists(np_) or args.force:
                    nl = run_null(model, tok, Xs[L], y, pool, L, alphas,
                                  args.bs, dtype, args.pairs_per_seed, args.n_rand)
                    with open(np_, "w") as f:
                        json.dump(nl, f, indent=2)
                    n8 = nl["alphas"][str(alphas[-1])]
                    print(f"    L{L:<3} NULL   rand A={n8['antisym_mean']:+.3f}"
                          f"+-{n8['antisym_std']:.3f}", flush=True)
            del Xs; gc.collect()

        del model, tok; gc.collect(); torch.mps.empty_cache()
    print(f"\ndone. cells in {CKPT}")


if __name__ == "__main__":
    main()
