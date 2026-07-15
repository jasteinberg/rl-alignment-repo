"""
scripts/check_rogue_dimension.py

Is the large steering effect on `counterfact` a truth direction, or the model's
massive-activation ("rogue") dimension?

Diagnosis. At pythia-2.8b layer 20 the within-class covariance of counterfact
activations has lambda_1 / tr(Sigma) = 0.997 -- ONE eigenvalue holds 99.7% of the
variance -- and |cos(theta_hat, v_1)| = 1.000. The mean shift is weak
(||delta|| = 9.5 vs sqrt(tr Sigma) = 107), so the difference-in-means estimator has
no truth signal to lock onto and collapses onto the highest-variance direction.
Steering then injects alpha * sqrt(lambda_1) along the model's most influential
axis. The output moves; truth has nothing to do with it.

Arms, each a unit vector u steered as alpha * std(X @ u) * u ("alpha sigmas along u"):
  theta       mass-mean direction (what the study used)
  v1          top eigenvector of the within-class covariance -- NO LABELS USED
  theta_perp  mass-mean refit after projecting v_1 out of the activations
  random      matched, meaningless

Predictions registered BEFORE running:
  counterfact: A(v1) ~= A(theta) ~= +1.0..1.1  (they are the same vector)
               A(theta_perp) ~= 0              => effect was entirely the rogue dim
  cities:      A(v1) < A(theta), A(theta_perp) retains most of A(theta)
               (|cos(theta,v1)| = 0.331 there; only 6.9% of variance along theta)

Falsifier: A(v1) ~= 0 on counterfact while A(theta) ~= +1.09, despite cos = 1.000.
That would be internally contradictory, and would rescue the original claim.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, gc, argparse, importlib.util
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c2", os.path.join(REPO, "scripts/steer_confirm2.py"))
C2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(C2)
S, SC = C2.S, C2.SC
ART, DEV = os.path.join(REPO, "artifacts"), "mps"


def spectrum(X, y):
    """Within-class covariance spectrum + how the mass-mean direction sits in it."""
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] -= X[y == lab].mean(0)
    C = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(C)
    w = w[::-1]; V = V[:, ::-1]
    v1 = V[:, 0]
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)
    if th @ v1 < 0:           # v1's sign is arbitrary; orient it with theta
        v1 = -v1
    return {
        "lambda1_over_trace": float(w[0] / w.sum()),
        "lambda1": float(w[0]), "lambda2": float(w[1]),
        "cos_theta_v1": float(abs(th @ v1)),
        "var_along_theta_frac": float((th @ C @ th) / w.sum()),
        "delta_over_sqrt_trace": float(np.linalg.norm(delta) / np.sqrt(w.sum())),
    }, v1, C


def build_arms(X, y, seed=0):
    """theta, v1 (label-free), theta_perp (after removing v1), random."""
    tr, te = C2.S.split_indices(y, seed=seed)
    _, v1, _ = spectrum(X[tr], y[tr])

    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th

    Xp = X - np.outer(X @ v1, v1)          # project out the rogue dimension
    thp = S.mass_mean_direction(Xp[tr], y[tr])
    if S.auroc(Xp[tr] @ thp, y[tr]) < 0.5: thp = -thp

    rng = np.random.default_rng(4242 + seed)
    r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)

    # orient v1 for steering the same way theta is oriented
    if S.auroc(X[tr] @ v1, y[tr]) < 0.5: v1 = -v1

    arms = {"theta": (th, X), "v1": (v1, X), "theta_perp": (thp, Xp), "random": (r, X)}
    aur = {k: S.auroc(Xa[te] @ u, y[te]) for k, (u, Xa) in arms.items()}
    return arms, aur, te


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--datasets", default="counterfact_true_false,cities")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ART, "rogue_dimension.json"))
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    tok, model = S.get_model(args.model, DEV, torch.float16)
    dtype = next(model.parameters()).dtype
    nL = model.config.num_hidden_layers
    layers = sorted({max(1, int(round(f * nL))) for f in C2.SEED_PLAN})
    results = {}

    for ds in [d for d in args.datasets.split(",") if d]:
        Xs, y = C2.get_acts(model, tok, args.model, ds, layers, args.cap)
        pool = SC.load_pairs(ds, args.pairs, 0)
        results[ds] = {}
        print(f"\n{'='*78}\n{args.model} / {ds}\n{'='*78}", flush=True)

        for L in layers:
            X = Xs[L].astype(np.float64)
            sp, _, _ = spectrum(X, y)
            print(f"\n--- layer {L} ---", flush=True)
            print(f"  lambda1/tr={sp['lambda1_over_trace']:.4f}  "
                  f"lambda1={sp['lambda1']:.1f} lambda2={sp['lambda2']:.1f}  "
                  f"|cos(theta,v1)|={sp['cos_theta_v1']:.3f}  "
                  f"||delta||/sqrt(tr)={sp['delta_over_sqrt_trace']:.4f}", flush=True)

            per_arm = {k: {str(a): [] for a in alphas} for k in
                       ("theta", "v1", "theta_perp", "random")}
            aurocs = {k: [] for k in per_arm}

            for sd in seeds:
                arms, aur, te = build_arms(X, y, seed=sd)
                pairs = C2.pairs_for_seed(pool, sd, args.pairs_per_seed)
                for k in aurocs: aurocs[k].append(aur[k])
                with SC.Steerer(model, L) as st:
                    st.set(None, 0, DEV, dtype)
                    base = SC.score_pairs(model, tok, pairs, args.bs)
                    # norm-matched: every arm displaced by alpha*||delta||.
                    # Scaling each arm by its own std(X@u) gave v1 a push of
                    # sqrt(lambda_1) ~ 104 and theta_perp a tiny one -- the arms
                    # were not comparable.
                    sc = C2.class_gap(X, y)
                    for name, (u, Xa) in arms.items():
                        for a in alphas:
                            A, _ = C2.antisym(model, tok, st, u, a, sc,
                                              pairs, base, args.bs, dtype)
                            per_arm[name][str(a)].append(A)
                    st.set(None, 0, DEV, dtype)

            a_top = str(alphas[-1])
            print(f"  {'arm':<12} {'AUROC':>7} {'A(a=%s)'%a_top:>18}", flush=True)
            for name in ("theta", "v1", "theta_perp", "random"):
                v = np.array(per_arm[name][a_top])
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                print(f"  {name:<12} {np.mean(aurocs[name]):>7.3f} "
                      f"{v.mean():>+11.3f}+-{se:.3f}", flush=True)

            results[ds][str(L)] = {"spectrum": sp, "arms": per_arm,
                                   "auroc": {k: float(np.mean(v)) for k, v in aurocs.items()}}
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        del Xs; gc.collect()

    del model, tok; gc.collect(); torch.mps.empty_cache()
    print(f"\nwrote {args.out}\ndone.")


if __name__ == "__main__":
    main()
