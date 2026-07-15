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
import os, json, gc, glob, argparse, importlib.util
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(REPO, "scripts/snr_sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
spec2 = importlib.util.spec_from_file_location("sc", os.path.join(REPO, "scripts/steer_completions.py"))
SC = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(SC)

ART = os.path.join(REPO, "artifacts")
CKPT = os.path.join(ART, "steer_ckpt")  # class-gap alpha units
ACTS = os.path.join(ART, "act_cache")
DEV = "mps"

SEED_PLAN = {0.25: 10, 0.375: 10, 0.5: 10, 0.625: 5, 0.75: 5, 0.875: 10}


def tag(m): return m.split("/")[-1]


def cell_path(model, ds, layer, seed):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__s{seed}.json")


def null_path(model, ds, layer):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__NULL.json")


def act_path(model, ds):
    return os.path.join(ACTS, f"{tag(model)}__{ds}.npz")


def get_acts(model, tok, mname, ds, layers, cap, seed=0):
    """Cache last-token activations for the probed layers only.

    Full A is (n_layers+1, N, d) and is large; we keep just the probed layers,
    so extending the seed list later costs no forward passes at all.
    """
    p = act_path(mname, ds)
    if os.path.exists(p):
        z = np.load(p)
        if sorted(int(k[1:]) for k in z.files if k.startswith("L")) == sorted(layers):
            print(f"    (cached activations {os.path.basename(p)})", flush=True)
            return {int(k[1:]): z[k] for k in z.files if k.startswith("L")}, z["y"]
    stmts, y = S.load_dataset(ds, cap=cap, seed=seed)
    print(f"    extracting {len(stmts)} activations...", flush=True)
    A = S.extract_all_layers(stmts, tok, model, DEV, 16)
    out = {L: A[L].astype(np.float32) for L in layers}
    os.makedirs(ACTS, exist_ok=True)
    np.savez_compressed(p, y=y, **{f"L{L}": v for L, v in out.items()})
    del A; gc.collect()
    return out, y


def load_cells(art=ART):
    """Notebook entry point: every cell on disk as a tidy list of dicts.

    Robust to however many seeds happen to exist -- add more by re-running with
    --seeds and calling this again.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(art, "steer_ckpt", "*__s*.json"))):
        r = json.load(open(f))
        base = os.path.basename(f)[:-5].split("__")
        r["model"], r["dataset"] = base[0], base[1]
        rows.append(r)
    nulls = {}
    for f in sorted(glob.glob(os.path.join(art, "steer_ckpt", "*__NULL.json"))):
        base = os.path.basename(f)[:-5].split("__")
        nulls[(base[0], base[1], int(base[2][1:]))] = json.load(open(f))
    return rows, nulls


def antisym(model, tok, st, vec, alpha, scale, pairs, base, bs, dtype):
    st.set(vec, +alpha * scale, DEV, dtype)
    sp = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
    st.set(vec, -alpha * scale, DEV, dtype)
    sm = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
    return 0.5 * (sp - sm), 0.5 * (sp + sm)


def fit_dirs(X, y, seed):
    tr, te = S.split_indices(y, seed=seed)
    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    try:
        thw = S.whitened_direction(X[tr], y[tr])
        if S.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    except Exception:
        thw = th.copy()
    return th, thw, tr, te


def pairs_for_seed(pool, seed, n):
    """Stable given the seed: seed k always selects the same evaluation subset."""
    rng = np.random.default_rng(777 + seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def class_gap(X, y):
    """||delta|| = || mu_1 - mu_0 ||, the distance between the class means.

    This -- not std(X @ theta) -- is the right unit for a steering magnitude.
    It is a property of the DATASET AND LAYER, not of the direction, so every
    arm gets a norm-matched push and every layer is comparable. alpha = 1 means
    "displace by exactly the class-mean gap": along theta that carries the
    false-class mean onto the true-class mean.

    In sigma units the same alpha meant wildly different things: on counterfact
    alpha=8 was 90x the class separation at every layer, and on cities it ranged
    from 71x (layer 8) to 2.5x (layer 28) -- so a "depth profile of causal
    efficacy" was really a depth profile of how far off-manifold we pushed.
    """
    return float(np.linalg.norm(X[y == 1].mean(0) - X[y == 0].mean(0)))


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
           "probe_auroc": S.auroc(X[te] @ th, y[te]),
           "probe_auroc_whitened": S.auroc(X[te] @ thw, y[te]), "alphas": {}}
    with SC.Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = SC.score_pairs(model, tok, pairs, bs)
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
    with SC.Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = SC.score_pairs(model, tok, pairs, bs)
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
        tok, model = S.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        nL = model.config.num_hidden_layers
        plan = {max(1, int(round(f * nL))): n for f, n in SEED_PLAN.items()}
        print(f"\n=== {mname} ({nL} layers) plan={plan} ===", flush=True)

        for ds in [d for d in args.datasets.split(",") if d]:
            print(f"  [{ds}]", flush=True)
            layers = sorted(plan)
            Xs, y = get_acts(model, tok, mname, ds, layers, args.cap)
            pool = SC.load_pairs(ds, args.pairs, 0)

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
