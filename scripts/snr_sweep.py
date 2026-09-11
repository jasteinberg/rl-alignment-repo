"""
scripts/snr_sweep.py

Truth-direction SNR sweep across the Pythia ladder.

For each (model, dataset): one forward pass, last-token residual activations
cached for all layers; per layer, plain and whitened (Ledoit-Wolf) mass-mean
d', AUROC, accuracy, and a random-direction null (p50/p95).

At each model's best layer: cross-dataset transfer matrix, and a superposition
probe (d' vs number of top-k PCs removed).

Cover / N-sweep control on counterfact_true_false at full size: true vs
shuffled labels as N crosses the separating capacity 2d.

Writes artifacts/snr_sweep.json; checkpoints per (model, dataset) under
artifacts/ckpt/ so an interrupted run resumes.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse, json, os, sys, gc
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.env import ENV
from truthlib.data import (MAIN_TIER, SMALL_TIER, DISTRACTOR, MAIN_CAP, SMALL_CAP,
                           cap_for, load_dataset)
from truthlib.estimators import (d_prime, auroc, split_indices,
                                 evaluate_direction, mass_mean as mass_mean_direction,
                                 fisher as whitened_direction)
from truthlib.acts import get_model, extract_all_layers

DATA = Path(ENV.GOT)
ART = REPO / "artifacts"
CKPT = ART / "ckpt"

# ---- configuration -------------------------------------------------------
# Main tier capped at its smallest member (companies_true_false, N=1199) so
# that d', transfer, and N/d are directly comparable across datasets.

COVER_DATASET = "counterfact_true_false"           # full size N=31964
COVER_NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 31964]

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]

N_NULL = 200        # random directions for the null
PCA_KS = [0, 1, 2, 4, 8, 16, 32, 64]   # superposition probe


def fit_eval_split(X, y, kind="plain", seed=0):
    """Fit the direction on train, evaluate d'/AUROC on held-out test.

    The mass-mean estimator has little capacity, but with d comparable to N the
    estimated mean difference absorbs O(sqrt(d/N)) of noise; scoring in-sample
    then rewards exactly the fluctuations that defined the direction. A random
    direction gets no such bonus, so an in-sample d' would overstate how far the
    truth direction clears its null -- the very comparison this study rests on.

    The direction's sign is fixed on TRAIN (so that train AUROC >= 0.5) and then
    held fixed on test. d' is sign-blind, but AUROC is not: without this the
    held-out AUROC can land below 0.5 for a direction that genuinely separates,
    and any d'-based layer selection would be blind to the flip.
    """
    tr, te = split_indices(y, seed=seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    if kind == "plain":
        theta = mass_mean_direction(Xtr, ytr)
    else:
        theta = whitened_direction(Xtr, ytr)
    if auroc(Xtr @ theta, ytr) < 0.5:      # orient on train only
        theta = -theta
    return evaluate_direction(theta, Xte, yte), (te, theta)

def random_direction_null(X, y, n_draws=N_NULL, seed=0):
    """Distribution of AUROC / d' over random unit directions.

    AUROC orientation is arbitrary for a random vector, so fold to >= 0.5:
    the null we care about is 'how much separation does an arbitrary direction
    achieve', not its sign.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    aus, dps = [], []
    for _ in range(n_draws):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        z = X @ u
        a = auroc(z, y)
        aus.append(max(a, 1 - a))
        dps.append(d_prime(z, y))
    aus, dps = np.array(aus), np.array(dps)
    return {"auroc_p50": float(np.percentile(aus, 50)),
            "auroc_p95": float(np.percentile(aus, 95)),
            "d_prime_p50": float(np.percentile(dps, 50)),
            "d_prime_p95": float(np.percentile(dps, 95))}


def superposition_probe(X, y, ks=PCA_KS, seed=0):
    """d' of the mass-mean direction after projecting out the top-k PCs.

    Tests whether the truth signal merely rides the highest-variance
    (most salient) subspace, or occupies its own low-variance directions.
    The PCA basis and the direction are both fit on train, applied to test.
    """
    tr, te = split_indices(y, seed=seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    out = []
    for k in ks:
        if k == 0:
            Atr, Ate = Xtr, Xte
        else:
            V = Vt[:k]                             # (k, d), fit on train only
            Atr = Xtr - (Xtr @ V.T) @ V
            Ate = Xte - (Xte @ V.T) @ V
        theta = mass_mean_direction(Atr, ytr)
        r = evaluate_direction(theta, Ate, yte)
        out.append({"k": int(k), **r})
    return out


def cover_n_sweep(X, y, ns, d_model, seed=0):
    """True vs shuffled labels as N crosses the separating capacity 2d.

    Cover: for N points in general position in R^d, nearly every dichotomy is
    linearly separable while N << 2d. A shuffled labeling therefore separates
    about as well as the true one at small N/d, and collapses toward chance as
    N grows past capacity -- while a real truth direction persists.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for N in ns:
        if N > len(y):
            continue
        idx = rng.choice(len(y), size=N, replace=False)
        Xs, ys = X[idx], y[idx]
        if len(np.unique(ys)) < 2:
            continue
        th_true = mass_mean_direction(Xs, ys)
        r_true = evaluate_direction(th_true, Xs, ys)

        y_shuf = rng.permutation(ys)
        th_shuf = mass_mean_direction(Xs, y_shuf)
        r_shuf = evaluate_direction(th_shuf, Xs, y_shuf)

        rows.append({"N": int(N), "d_model": int(d_model),
                     "N_over_2d": float(N / (2 * d_model)),
                     "true": r_true, "shuffled": r_shuf})
    return rows

# ---- activations ---------------------------------------------------------
def ckpt_path(model, dset):
    tag = model.split("/")[-1]
    return CKPT / f"{tag}__{dset}.json"


def analyze_dataset(A, y, layer_stride=1, seed=0):
    """A: (L, N, d) activations. Per-layer plain + whitened + null."""
    L = A.shape[0]
    layers = list(range(0, L, layer_stride))
    if layers[-1] != L - 1:
        layers.append(L - 1)
    rows = []
    for li in layers:
        X = A[li].astype(np.float64)
        plain, (te, _) = fit_eval_split(X, y, "plain", seed=seed)
        try:
            whit, _ = fit_eval_split(X, y, "whitened", seed=seed)
        except Exception as e:                    # ill-conditioned Sigma
            whit = {"auroc": None, "d_prime": None, "acc": None,
                    "error": str(e)[:80]}
        # null on the SAME held-out points, so the comparison is apples-to-apples
        null = random_direction_null(X[te], y[te], seed=seed + li)
        rows.append({"layer": int(li), "plain": plain,
                     "whitened": whit, "null": null})
        print(f"      L{li:>2}  AUROC={plain['auroc']:.3f}  d'={plain['d_prime']:.2f}"
              f"  (null p95={null['auroc_p95']:.3f})", flush=True)
    return rows


def best_layer(rows):
    """Layer whose held-out AUROC most exceeds its own random-direction null.

    AUROC is sign-aware (d' is not), so a layer where the fitted direction is
    anti-predictive on held-out data cannot win. Layer 0 is the embedding, not
    a transformer block: on templated statements the last-token embedding is
    near-constant, giving a degenerate d'; it is excluded from selection.
    """
    cand = [r for r in rows if r["layer"] > 0] or rows
    scored = [(r["plain"]["auroc"] - r["null"]["auroc_p95"], r["layer"])
              for r in cand]
    return max(scored)[1]


# ---- main sweep ----------------------------------------------------------
def run_model(mname, datasets, device, dtype, args):
    tag = mname.split("/")[-1]
    print(f"\n=== {mname} ===", flush=True)
    tok, model = get_model(mname, device, dtype)
    d_model = model.config.hidden_size

    per_dataset, acts_at_best = {}, {}
    for dset in datasets:
        cp = ckpt_path(mname, dset)
        cap = args.cap_override or cap_for(dset)
        stmts, y = load_dataset(dset, cap=cap, seed=args.seed)
        if args.smoke:
            stmts, y = stmts[:args.smoke], y[:args.smoke]

        print(f"  [{dset}] N={len(stmts)}  d={d_model}  N/2d={len(stmts)/(2*d_model):.2f}",
              flush=True)
        A = extract_all_layers(stmts, tok, model, device, args.batch_size)

        if cp.exists() and not args.force:
            print(f"    (resume) {cp.name}", flush=True)
            per_dataset[dset] = json.loads(cp.read_text())
        else:
            rows = analyze_dataset(A, y, layer_stride=args.layer_stride, seed=args.seed)
            bl = best_layer(rows)
            entry = {"n": len(y), "d_model": int(d_model), "layers": rows,
                     "best_layer": bl,
                     "superposition": superposition_probe(A[bl].astype(np.float64), y, seed=args.seed)}
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(entry, indent=2))
            per_dataset[dset] = entry
            print(f"    best layer {bl}; checkpointed", flush=True)

        bl = per_dataset[dset]["best_layer"]
        acts_at_best[dset] = (A[bl].astype(np.float64), y)
        del A; gc.collect()

    # transfer across the main tier. The direction is fit on each source's TRAIN
    # half and scored on every target's TEST half, so the diagonal is held-out
    # too and is directly comparable to the off-diagonal.
    tier = [d for d in datasets if d in MAIN_TIER]
    transfer = {}
    for src in tier:
        Xs, ys = acts_at_best[src]
        tr_s, _ = split_indices(ys, seed=args.seed)
        th = mass_mean_direction(Xs[tr_s], ys[tr_s])
        if auroc(Xs[tr_s] @ th, ys[tr_s]) < 0.5:   # orient on source train
            th = -th
        row = {}
        for tgt in tier:
            Xt, yt = acts_at_best[tgt]
            _, te_t = split_indices(yt, seed=args.seed)
            row[tgt] = evaluate_direction(th, Xt[te_t], yt[te_t])["auroc"]
        transfer[src] = row

    # Cover control at full size on the designated dataset
    cover = None
    if args.cover and COVER_DATASET in datasets:
        print(f"  [cover] {COVER_DATASET} full size", flush=True)
        stmts, y = load_dataset(COVER_DATASET, cap=None, seed=args.seed)
        if args.smoke:
            stmts, y = stmts[:args.smoke], y[:args.smoke]
        bl = per_dataset[COVER_DATASET]["best_layer"]
        A = extract_all_layers(stmts, tok, model, device, args.batch_size)
        ns = [n for n in COVER_NS if n <= len(y)] or [len(y)]
        cover = cover_n_sweep(A[bl].astype(np.float64), y, ns, d_model, seed=args.seed)
        for r in cover:
            print(f"      N={r['N']:>6} N/2d={r['N_over_2d']:.2f}  "
                  f"true AUROC={r['true']['auroc']:.3f}  "
                  f"shuf AUROC={r['shuffled']['auroc']:.3f}", flush=True)
        del A; gc.collect()

    del model, tok; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {"d_model": int(d_model), "datasets": per_dataset,
            "transfer": transfer, "cover": cover}


# ---- cli -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(MAIN_TIER + SMALL_TIER + DISTRACTOR))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--layer_stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap_override", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0,
                    help="truncate every dataset to this many rows")
    ap.add_argument("--no-cover", dest="cover", action="store_false")
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    ap.add_argument("--out", default=str(ART / "snr_sweep.json"))
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.device == "mps" else torch.float32
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    results = {}
    for mname in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            results[mname] = run_model(mname, datasets, args.device, dtype, args)
        except Exception as e:
            print(f"  !! {mname} failed: {type(e).__name__}: {e}", flush=True)
            results[mname] = {"error": f"{type(e).__name__}: {e}"}
        # write after every model so a later crash cannot lose earlier rungs
        Path(args.out).write_text(json.dumps(
            {"config": {"main_tier": MAIN_TIER, "small_tier": SMALL_TIER,
                        "distractor": DISTRACTOR, "main_cap": MAIN_CAP,
                        "small_cap": SMALL_CAP, "n_null": N_NULL,
                        "pca_ks": PCA_KS, "cover_dataset": COVER_DATASET,
                        "seed": args.seed},
             "models": results}, indent=2))
        print(f"  -> wrote {args.out}", flush=True)

    print("\ndone.")


if __name__ == "__main__":
    main()
