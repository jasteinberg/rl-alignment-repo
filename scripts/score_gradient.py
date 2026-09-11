"""scripts/score_gradient.py

Measures g = <sum_t grad_{x_t} ell>, the mean gradient of the behavioural score
with respect to the layer-L residual stream, summed over token positions.

Why the position sum: the steering hook (steer_completions.Steerer) adds the
same vector at EVERY position, so the linear response of the score to a push
along w is  d(ell)/dh = c * (w . sum_t grad_{x_t} ell).  Differentiating only
the final token would not be the quantity the steering arms measure.

ell is taken from steer_completions.score_pairs by construction: same
contrastive pairs, same tokenisation, same summation over completion tokens,
same pool/seed selection. The score is a DIFFERENCE of two forward passes over
different sequences, so the gradient is likewise the difference of the two
per-sequence gradients.

Outputs cos(g, .) against the mass-mean, whitened, rogue and orthogonal-gap
directions, plus the predicted small-h susceptibility chi_lin(w) = scale*(w.g)
for comparison against the measured chi (a through-origin fit over
alpha in {0.5,1,2,4}, hence NOT a pure h->0 derivative).

Drafted with the assistance of Claude (Anthropic).
"""
import os, sys, json, gc, argparse
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from truthlib import acts, data, steering
from truthlib import estimators as est

ART = os.path.join(REPO, "artifacts")
DEV = "mps"
LOSS_SCALE = 1024.0          # fp16 backward underflows without this


def rogue_and_gap(X, y, tr):
    """v1 (leading eigenvector of the within-class covariance) and e2."""
    Xtr = X[tr]; ytr = y[tr]
    Xc = np.vstack([Xtr[ytr == k] - Xtr[ytr == k].mean(0) for k in (0, 1)])
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    v1 = V[:, -1]
    lam = w[::-1]
    delta = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    d_perp = delta - (v1 @ delta) * v1
    e2 = d_perp / np.linalg.norm(d_perp)
    return v1, e2, lam


def grad_for_texts(model, tok, block, texts, n_prompts, n_fulls):
    """sum_t d(sum log P(completion tokens)) / d(layer-L residual at t) -> (B, d)."""
    store = {}

    def hook(mod, inp, out):
        # Replace the block output with a leaf that requires grad: downstream
        # layers then depend on it, .grad lands on exactly the tensor the
        # Steerer adds to, and backprop stops here rather than continuing to
        # the embeddings.
        h = out[0] if isinstance(out, tuple) else out
        h2 = h.detach().requires_grad_(True)
        store["h"] = h2
        return (h2,) + out[1:] if isinstance(out, tuple) else h2

    handle = block.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            logits = model(**enc).logits.float().log_softmax(-1)
            loss = 0.0
            for j in range(len(texts)):
                for t in range(n_prompts[j], n_fulls[j]):
                    loss = loss + logits[j, t - 1, enc["input_ids"][j, t]]
            (LOSS_SCALE * loss).backward()
            g = store["h"].grad.detach().float().sum(1) / LOSS_SCALE   # (B, d)
            return g.cpu().numpy().astype(np.float64)
    finally:
        handle.remove()
        store.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    ap.add_argument("--layers", default="28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=400)        # pool, as in steer_confirm2
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ART, "score_gradient.json"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    tok, model = acts.get_model(args.model, DEV, torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stmts, y = data.load_dataset(args.dataset, cap=args.cap, seed=args.seed)
    print(f"extracting activations ({len(stmts)} statements)...", flush=True)
    A = acts.extract_all_layers(stmts, tok, model, DEV, 16)

    pool = data.load_pairs(args.dataset, args.pairs, 0)
    pairs = steering.pairs_for_seed(pool, args.seed, args.pairs_per_seed)
    print(f"{len(pairs)} contrastive pairs\n", flush=True)

    out = {"model": args.model, "dataset": args.dataset, "seed": args.seed,
           "n_pairs": len(pairs), "layers": {}}

    for L in layers:
        X = A[L].astype(np.float64)
        tr, te = est.split_indices(y, seed=args.seed)
        th, thw, _, _ = steering.fit_dirs(X, y, args.seed)
        v1, e2, lam = rogue_and_gap(X, y, tr)
        scale = est.class_gap(X[tr], y[tr])
        block = model.gpt_neox.layers[L]

        gs = []
        for i in range(0, len(pairs), args.bs):
            chunk = pairs[i:i + args.bs]
            per_side = {}
            for which in ("true", "false"):
                texts = [p["prompt"] + p[which] for p in chunk]
                n_prompts = [len(tok(p["prompt"]).input_ids) for p in chunk]
                n_fulls = [len(tok(p["prompt"] + p[which]).input_ids) for p in chunk]
                per_side[which] = grad_for_texts(model, tok, block, texts,
                                                 n_prompts, n_fulls)
            gs.append(per_side["true"] - per_side["false"])
            if (i // args.bs) % 10 == 0:
                print(f"  L{L}  {i + len(chunk)}/{len(pairs)}", flush=True)
        G = np.vstack(gs)                      # (n_pairs, d)
        g = G.mean(0)
        gn = float(np.linalg.norm(g))

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        rec = {
            "alpha_unit": scale,
            "lam1_over_tr": float(lam[0] / lam.sum()),
            "lam1_over_lam2": float(lam[0] / lam[1]),
            "g_norm": gn,
            "cos_g_v1": cos(g, v1),
            "cos_g_theta": cos(g, th),
            "cos_g_theta_whitened": cos(g, thw),
            "cos_g_e2": cos(g, e2),
            "cos_theta_v1": cos(th, v1),
            "plane_fraction": float(np.sqrt((g @ v1) ** 2 + (g @ e2) ** 2) / gn),
            "chi_lin": {
                "plain": scale * float(th @ g),
                "whitened": scale * float(thw @ g),
                "v1": scale * float(v1 @ g),
                "e2": scale * float(e2 @ g),
                "g_ceiling": scale * gn,
            },
        }
        out["layers"][str(L)] = rec
        print(json.dumps(rec, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        # Save the vectors themselves: the summary above answers the part-1
        # question, but g projected on the eigenbasis, the steering ceiling,
        # and any audit of a published direction all need g, v1, e2 and the
        # spectrum, not their pairwise cosines.
        vec_path = args.out.replace(".json", "_vectors.npz")
        prev = dict(np.load(vec_path)) if os.path.exists(vec_path) else {}
        prev.update({f"L{L}_g": g, f"L{L}_g_per_pair": G.astype(np.float32),
                     f"L{L}_v1": v1, f"L{L}_e2": e2, f"L{L}_theta": th,
                     f"L{L}_theta_whitened": thw, f"L{L}_lam": lam,
                     f"L{L}_scale": np.array([scale])})
        np.savez_compressed(vec_path, **prev)
        print(f"  vectors -> {vec_path}", flush=True)
        gc.collect()

    print(f"\nwrote {args.out}\ndone.")


if __name__ == "__main__":
    main()
