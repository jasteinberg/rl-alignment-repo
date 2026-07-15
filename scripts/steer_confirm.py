"""
scripts/steer_confirm.py

Confirmatory run for the decodability-vs-causality dissociation.

The single-seed sweep found that steering efficacy on `cities` peaks at layer 12
(probe AUROC 0.591) and vanishes by layer 28 (probe AUROC 0.978). Before that is
a claim, it needs: multiple split seeds, a second dataset (counterfact, where the
direction is BURIED beneath the salient subspace), and a second model.

Method note. A steering effect must be decomposed:

    shift(+a) and shift(-a)
    antisymmetric  A = [shift(+a) - shift(-a)] / 2   <- real steering
    symmetric      Sy = [shift(+a) + shift(-a)] / 2  <- perturbation damage

Only A is evidence of a causal direction: a generic out-of-distribution
perturbation degrades the forward pass for BOTH signs, contributing to Sy while
leaving A near zero. Reporting a raw shift, or |z| against a random control,
conflates the two -- at layers 16/20 the single-seed run showed |z| up to 38 with
A ~ 0, i.e. pure damage.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, gc, argparse, importlib.util
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(REPO, "scripts/snr_sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
spec2 = importlib.util.spec_from_file_location("sc", os.path.join(REPO, "scripts/steer_completions.py"))
SC = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(SC)

ART = os.path.join(REPO, "artifacts")
DEV = "mps"


def probe_and_steer(model, tok, A, y, pairs, layer, alphas, seed, n_rand, bs, dtype):
    """Fit directions at `layer` on split `seed`; return sym/antisym decomposition."""
    X = A[layer].astype(np.float64)
    tr, te = S.split_indices(y, seed=seed)

    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    try:
        thw = S.whitened_direction(X[tr], y[tr])
        if S.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    except Exception:
        thw = th.copy()

    rng = np.random.default_rng(5000 + seed * 31 + layer)
    rands = []
    for _ in range(n_rand):
        r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
        rands.append(r)

    scale = float(np.std(X @ th))
    auroc_held = S.auroc(X[te] @ th, y[te])
    aurocw_held = S.auroc(X[te] @ thw, y[te])

    out = {"layer": layer, "seed": seed, "probe_auroc": auroc_held,
           "probe_auroc_whitened": aurocw_held, "alpha_unit": scale, "alphas": {}}

    with SC.Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = SC.score_pairs(model, tok, pairs, bs)
        for a in alphas:
            rec = {}
            for name, vec in (("plain", th), ("whitened", thw)):
                st.set(vec, +a * scale, DEV, dtype)
                sp = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
                st.set(vec, -a * scale, DEV, dtype)
                sm = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
                rec[name] = {"shift_pos": sp, "shift_neg": sm,
                             "antisym": 0.5 * (sp - sm), "sym": 0.5 * (sp + sm)}
            ra = []
            for r in rands:
                st.set(r, +a * scale, DEV, dtype)
                sp = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
                st.set(r, -a * scale, DEV, dtype)
                sm = float(np.mean(SC.score_pairs(model, tok, pairs, bs) - base))
                ra.append(0.5 * (sp - sm))
            rec["random"] = {"antisym_mean": float(np.mean(ra)),
                             "antisym_std": float(np.std(ra))}
            out["alphas"][str(a)] = rec
            print(f"      a={a:>4}  plain A={rec['plain']['antisym']:+6.3f} "
                  f"S={rec['plain']['sym']:+6.3f} | whit A={rec['whitened']['antisym']:+6.3f} "
                  f"| rand A={rec['random']['antisym_mean']:+6.3f}+-{rec['random']['antisym_std']:.3f}",
                  flush=True)
        st.set(None, 0, DEV, dtype)
    out["baseline"] = float(base.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b,EleutherAI/pythia-1.4b")
    ap.add_argument("--datasets", default="cities,counterfact_true_false")
    ap.add_argument("--layer_frac", default="0.25,0.375,0.5,0.625,0.75,0.875",
                    help="layers as fractions of depth, so models are comparable")
    ap.add_argument("--alphas", default="2,4,8")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ART, "steer_confirm.json"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.layer_frac.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    results = {}

    for mname in [m for m in args.models.split(",") if m]:
        tok, model = S.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        n_layers = model.config.num_hidden_layers
        layers = sorted({max(1, int(round(f * n_layers))) for f in fracs})
        print(f"\n=== {mname}  ({n_layers} layers -> probing {layers}) ===", flush=True)
        results[mname] = {"n_layers": n_layers, "datasets": {}}

        for ds in [d for d in args.datasets.split(",") if d]:
            stmts, y = S.load_dataset(ds, cap=args.cap, seed=0)
            print(f"  [{ds}] extracting {len(stmts)} activations...", flush=True)
            A = S.extract_all_layers(stmts, tok, model, DEV, 16)
            pairs = SC.load_pairs(ds, args.pairs, 0)
            print(f"  [{ds}] {len(pairs)} contrastive pairs", flush=True)
            rows = []
            for L in layers:
                for sd in seeds:
                    print(f"    layer {L} seed {sd}", flush=True)
                    rows.append(probe_and_steer(model, tok, A, y, pairs, L,
                                                alphas, sd, args.n_rand, args.bs, dtype))
            results[mname]["datasets"][ds] = rows
            del A; gc.collect()
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  -> wrote {args.out}", flush=True)

        del model, tok; gc.collect(); torch.mps.empty_cache()
    print("\ndone.")


if __name__ == "__main__":
    main()
