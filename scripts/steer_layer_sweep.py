"""
scripts/steer_layer_sweep.py

Feasibility: is there ANY (layer, alpha) at which steering along the truth
direction produces a *truth-specific* behavioural effect on Pythia-2.8b?

Two failure modes bracket the search:
  - alpha too large -> the added vector is out-of-distribution, the forward pass
    degrades, and a RANDOM direction hurts just as much (no specificity).
  - alpha too small, or intervention too late -> nothing propagates, and the
    effect is indistinguishable from the random control (no effect).

So the quantity that matters is not the shift but the SPECIFICITY MARGIN:

    margin(layer, alpha) = shift[plain] - shift[random]

with both directions at matched norm. A causal truth direction must show a
margin that (i) exceeds zero, (ii) is antisymmetric under alpha -> -alpha, and
(iii) exceeds the seed-to-seed spread of the random control.

Readout: log P(true completion) - log P(false completion) on `cities`, the one
dataset where pythia-2.8b demonstrably has a behaviour (behavioural AUROC 0.823).
Steering is applied at ALL token positions so the perturbation can propagate.

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


def directions_at_layer(A, y, layer, n_rand=3, seed=0):
    """Fit plain + whitened at `layer`; return them plus n_rand random controls.

    The alpha unit is the std of the projection onto the plain direction at this
    layer, so alpha is comparable across layers ('move one sigma along theta').
    """
    X = A[layer].astype(np.float64)
    tr, _ = S.split_indices(y, seed=seed)
    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    try:
        thw = S.whitened_direction(X[tr], y[tr])
        if S.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    except Exception:
        thw = th.copy()
    rng = np.random.default_rng(1000 + seed + layer)
    rands = []
    for _ in range(n_rand):
        r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
        rands.append(r)
    scale = float(np.std(X @ th))
    auroc_here = S.auroc(X @ th, y)
    return th, thw, rands, scale, auroc_here


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dataset", default="cities")
    ap.add_argument("--layers", default="8,12,16,20,24,28")
    ap.add_argument("--alphas", default="-8,-4,-2,2,4,8")
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ART, "steer_layer_sweep.json"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    tok, model = S.get_model(args.model, DEV, torch.float16)
    dtype = next(model.parameters()).dtype

    stmts, y = S.load_dataset(args.dataset, cap=args.cap, seed=args.seed)
    print(f"extracting activations ({len(stmts)} statements)...", flush=True)
    A = S.extract_all_layers(stmts, tok, model, DEV, 16)

    pairs = SC.load_pairs(args.dataset, args.pairs, args.seed)
    print(f"{len(pairs)} contrastive pairs\n", flush=True)

    results = []
    for L in layers:
        th, thw, rands, scale, au = directions_at_layer(A, y, L, args.n_rand, args.seed)
        print(f"=== layer {L}  (probe AUROC={au:.3f}, alpha unit={scale:.2f}) ===", flush=True)
        with SC.Steerer(model, L) as st:
            st.set(None, 0, DEV, dtype)
            base = SC.score_pairs(model, tok, pairs, args.bs)
            print(f"    baseline mean score = {base.mean():+.3f}", flush=True)
            for a in alphas:
                st.set(th, a * scale, DEV, dtype)
                sp = SC.score_pairs(model, tok, pairs, args.bs)
                shift_p = float(np.mean(sp - base))
                st.set(thw, a * scale, DEV, dtype)
                sw = SC.score_pairs(model, tok, pairs, args.bs)
                shift_w = float(np.mean(sw - base))
                rs = []
                for r in rands:
                    st.set(r, a * scale, DEV, dtype)
                    sr = SC.score_pairs(model, tok, pairs, args.bs)
                    rs.append(float(np.mean(sr - base)))
                rmean, rstd = float(np.mean(rs)), float(np.std(rs))
                margin = shift_p - rmean
                z = margin / rstd if rstd > 1e-9 else float("nan")
                print(f"    a={a:+5.1f}  plain={shift_p:+7.3f}  whit={shift_w:+7.3f}  "
                      f"rand={rmean:+7.3f}+-{rstd:.3f}  margin={margin:+7.3f}  z={z:+5.1f}",
                      flush=True)
                results.append({"layer": L, "alpha": a, "probe_auroc": au,
                                "shift_plain": shift_p, "shift_whitened": shift_w,
                                "rand_mean": rmean, "rand_std": rstd,
                                "margin": margin, "z": z,
                                "baseline": float(base.mean())})
            st.set(None, 0, DEV, dtype)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "dataset": args.dataset,
                       "rows": results}, f, indent=2)
        gc.collect()
    print(f"\nwrote {args.out}\ndone.")


if __name__ == "__main__":
    main()
