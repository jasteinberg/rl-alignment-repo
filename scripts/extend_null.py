"""
scripts/extend_null.py

The random-direction null for steering is currently estimated from 6 draws, so
its spread (+-0.2 to +-0.9) is far too uncertain to support significance claims.
This extends it to n_rand draws per (model, dataset, layer, alpha) and reports a
p95, matching how the probe null is reported elsewhere in the study.

Draws are NOT seeds. A seed varies the train/test split, hence the fitted theta;
averaging over seeds gives the standard error on the truth direction's effect. A
draw varies the random control vector -- no fitting is involved, so draws do not
depend on the split. What we want from the null is the *distribution* of the
antisymmetric effect produced by meaningless directions, and specifically its
p95: the effect size a random direction exceeds only 5% of the time.

Existing draws are reused; only the shortfall is computed. Results merge into the
same NULL files, so this is safe to re-run and to interrupt.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, gc, glob, argparse
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from truthlib import acts, data, steering
from truthlib import estimators as est
DEV = "mps"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b,EleutherAI/pythia-1.4b")
    ap.add_argument("--datasets", default="cities,counterfact_true_false")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--n_rand", type=int, default=30, help="TOTAL draws wanted")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--layers", default="", help="comma list; default all in SEED_PLAN")
    args = ap.parse_args()
    want = {int(s) for s in args.layers.split(",") if s}
    alphas = [float(a) for a in args.alphas.split(",")]

    for mname in [m for m in args.models.split(",") if m]:
        tok, model = acts.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        nL = model.config.num_hidden_layers
        layers = sorted({max(1, int(round(f * nL))) for f in steering.SEED_PLAN})
        if want:
            layers = [L for L in layers if L in want]
        print(f"\n=== {mname} ({nL} layers) -> {layers} ===", flush=True)

        for ds in [d for d in args.datasets.split(",") if d]:
            Xs, y = steering.get_acts(model, tok, mname, ds, layers, args.cap)
            pool = data.load_pairs(ds, args.pairs, 0)
            pairs = steering.pairs_for_seed(pool, 0, args.pairs_per_seed)

            for L in layers:
                p = steering.null_path(mname, ds, L)
                cur = json.load(open(p)) if os.path.exists(p) else {"layer": L, "alphas": {}}
                X = Xs[L].astype(np.float64)
                # Scale in class-gap units, matching run_null in steer_confirm2
                # ("same unit as the theta arms"). Note this is not std(X @ th),
                # which differs by sigma_along_theta and is layer-dependent.
                scale = est.class_gap(X, y)

                with steering.Steerer(model, L) as st:
                    st.set(None, 0, DEV, dtype)
                    base = steering.score_pairs(model, tok, pairs, args.bs)
                    for a in alphas:
                        k = str(a)
                        have = cur["alphas"].get(k, {}).get("draws", [])
                        need = args.n_rand - len(have)
                        if need <= 0:
                            print(f"    L{L} a={a}: already {len(have)} draws", flush=True)
                            continue
                        # offset the rng so new draws are independent of the old
                        rng = np.random.default_rng(9000 + L * 17 + 101 * len(have))
                        vals = list(have)
                        for _ in range(need):
                            r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
                            av, _ = steering.antisym(model, tok, st, r, a, scale,
                                               pairs, base, args.bs, dtype)
                            vals.append(av)
                            # Checkpoint every 10 draws so a long extension survives
                            # interruption. Mid-alpha resume reseeds the rng from the
                            # new len(have) -- draws stay independent; no consumer
                            # needs draw alignment across alphas (signflip reads the
                            # per-alpha p95, chi_whitening_analysis the alpha=1 draws).
                            if len(vals) % 10 == 0:
                                v = np.array(vals)
                                cur["alphas"][k] = {
                                    "antisym_mean": float(v.mean()),
                                    "antisym_std": float(v.std(ddof=1)),
                                    "antisym_p95": float(np.percentile(np.abs(v), 95)),
                                    "n_draws": len(v), "draws": [float(x) for x in v]}
                                with open(p, "w") as f:
                                    json.dump(cur, f, indent=2)
                                print(f"    L{L} a={a}: {len(vals)}/{args.n_rand} draws",
                                      flush=True)
                        v = np.array(vals)
                        cur["alphas"][k] = {
                            "antisym_mean": float(v.mean()),
                            "antisym_std": float(v.std(ddof=1)),
                            "antisym_p95": float(np.percentile(np.abs(v), 95)),
                            "n_draws": len(v), "draws": [float(x) for x in v]}
                        print(f"    L{L} a={a}: {len(v)} draws  mean={v.mean():+.3f} "
                              f"sd={v.std(ddof=1):.3f}  |A|p95={np.percentile(np.abs(v),95):.3f}",
                              flush=True)
                    st.set(None, 0, DEV, dtype)
                cur["n_rand"] = args.n_rand
                with open(p, "w") as f:
                    json.dump(cur, f, indent=2)
            del Xs; gc.collect()
        del model, tok; gc.collect(); torch.mps.empty_cache()
    print("\ndone.")


if __name__ == "__main__":
    main()
