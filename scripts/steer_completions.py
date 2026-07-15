"""
scripts/steer_completions.py

Causal test: does a truth direction found by probing actually move the model's
factual behaviour -- and does it matter whether the direction is SALIENT (lies in
the top principal components, e.g. `cities`) or BURIED (recovered only after the
salient subspace is removed, e.g. `counterfact`)?

Base LMs like Pythia do not reliably answer "This statement is: TRUE/FALSE"
(measured: behavioural AUROC 0.823 on cities but ~0.50 on counterfact). So the
readout is the model's *factual preference* over contrastive completions:

    prompt:  "The city of Krasnodar is in"
    score:   log P(" Russia") - log P(" South Africa")          [true - false]

This is a behaviour a base LM genuinely has. We add alpha * theta to the residual
stream at the probe's best layer and measure how the score shifts.

Controls:
  - matched-norm RANDOM direction (any perturbation moves logits; this bounds it)
  - plain vs whitened theta (whitened recovers buried signal; is it causal too?)
  - cross-dataset: steer with dataset A's direction, score dataset B

Drafted with the assistance of Claude (Anthropic).
"""
import os, sys, json, gc, argparse, importlib.util
import numpy as np, pandas as pd, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(REPO, "scripts/snr_sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

DATA = os.path.expanduser("~/external/geometry-of-truth/datasets")
ART = os.path.join(REPO, "artifacts")
DEV = "mps"


# ---- contrastive completion pairs ---------------------------------------
def pairs_cities(df, negated=False):
    """'The city of Krasnodar is in' -> (' Russia', ' South Africa').

    Each city appears with a true and a false country. We pull the true row to
    get correct_country and a false row to get a distractor.

    Under negation the stem becomes '... is not in', which INVERTS which
    completion makes a true sentence: 'Krasnodar is not in South Africa' is
    true, 'Krasnodar is not in Russia' is false. So the targets swap. (Without
    this the negated readout is sign-flipped and every steering result on
    neg_cities comes out backwards.)
    """
    out = []
    for city, g in df.groupby("city"):
        t = g[g["correct_country"] == g["country"]]
        f = g[g["correct_country"] != g["country"]]
        if len(t) == 0 or len(f) == 0:
            continue
        cc = t.iloc[0]["correct_country"]
        fc = f.iloc[0]["country"]
        if negated:
            stem = f"The city of {city} is not in"
            true_tgt, false_tgt = f" {fc}", f" {cc}"
        else:
            stem = f"The city of {city} is in"
            true_tgt, false_tgt = f" {cc}", f" {fc}"
        out.append({"prompt": stem, "true": true_tgt, "false": false_tgt})
    return out


def pairs_counterfact(df):
    """Uses the shipped relation template, subject, target, true_target."""
    out = []
    for subj, g in df.groupby("subject"):
        r = g.iloc[0]["relation"]
        tt = g.iloc[0]["true_target"]
        wrong = g[g["target"] != g["true_target"]]
        if len(wrong) == 0 or not isinstance(r, str) or "{}" not in r:
            continue
        ft = wrong.iloc[0]["target"]
        out.append({"prompt": r.format(subj).rstrip(),
                    "true": f" {tt}", "false": f" {ft}"})
    return out


BUILDERS = {
    "cities": lambda d: pairs_cities(d),
    "neg_cities": lambda d: pairs_cities(d, negated=True),
    "counterfact_true_false": pairs_counterfact,
}


def load_pairs(name, cap, seed=0):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    pr = BUILDERS[name](df)
    rng = np.random.default_rng(seed)
    if len(pr) > cap:
        idx = rng.choice(len(pr), size=cap, replace=False)
        pr = [pr[i] for i in idx]
    return pr


# ---- steering + scoring --------------------------------------------------
class Steerer:
    """Adds alpha * theta to the residual stream at `layer`, all positions.

    Pythia (GPTNeoX) block output is a tuple; we perturb element 0. Registering
    on layer L's block means the addition is visible to every later layer, which
    is where the probe direction was measured.
    """
    def __init__(self, model, layer):
        self.block = model.gpt_neox.layers[layer]
        self.vec = None
        self.h = None

    def __enter__(self):
        def hook(mod, inp, out):
            if self.vec is None:
                return out
            if isinstance(out, tuple):
                return (out[0] + self.vec.to(out[0].dtype),) + out[1:]
            return out + self.vec.to(out.dtype)
        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.h: self.h.remove()

    def set(self, theta, alpha, device, dtype):
        if theta is None or alpha == 0:
            self.vec = None
        else:
            t = torch.tensor(theta, device=device, dtype=dtype)
            self.vec = alpha * t


@torch.no_grad()
def score_pairs(model, tok, pairs, bs=8):
    """log P(true completion) - log P(false completion), summed over its tokens.

    Both completions are scored against the SAME prompt, so prompt length and
    any generic token bias cancel in the difference. Multi-token targets are
    summed (not length-normalised): the pair is scored on total log-prob, and
    length differences between the two targets are a property of the pair, not
    of the steering, so they cancel when we look at the *shift* under steering.
    """
    scores = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        vals = []
        for which in ("true", "false"):
            texts = [p["prompt"] + p[which] for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            logits = model(**enc).logits.float().log_softmax(-1)
            batch_vals = []
            for j, p in enumerate(chunk):
                n_prompt = len(tok(p["prompt"]).input_ids)
                n_full = len(tok(p["prompt"] + p[which]).input_ids)
                lp = 0.0
                for t in range(n_prompt, n_full):
                    tid = enc["input_ids"][j, t]
                    lp += logits[j, t - 1, tid].item()
                batch_vals.append(lp)
            vals.append(np.array(batch_vals))
        scores.append(vals[0] - vals[1])
    return np.concatenate(scores)


# ---- directions ----------------------------------------------------------
def fit_directions(model, tok, dset, layer, cap, seed=0):
    """Fit plain + whitened theta on the TRAIN half of `dset` at `layer`.

    Returns unit vectors, oriented so that higher projection = 'more true'.
    A matched-norm random direction is returned as the control.
    """
    stmts, y = S.load_dataset(dset, cap=cap, seed=seed)
    A = S.extract_all_layers(stmts, tok, model, DEV, 16)
    X = A[layer].astype(np.float64)
    tr, _ = S.split_indices(y, seed=seed)

    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    thw = S.whitened_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw

    rng = np.random.default_rng(seed + 99)
    rnd = rng.standard_normal(X.shape[1]); rnd /= np.linalg.norm(rnd)

    # alpha unit = std of the activations' projection ONTO the probe direction.
    # Using the full activation norm (||x|| ~ 1e2) would add a vector as large
    # as the entire residual stream: that is an out-of-distribution perturbation
    # which degrades the forward pass regardless of direction, and a random
    # control degrades it just as much. The projection scale is the natural unit
    # -- alpha=1 moves a statement one standard deviation along the direction.
    scale = float(np.std(X @ th))
    proj_stats = {"scale_proj_std": scale,
                  "mean_act_norm": float(np.linalg.norm(X, axis=1).mean()),
                  "class_sep": float(abs((X[y == 1] @ th).mean() - (X[y == 0] @ th).mean()))}
    del A; gc.collect()
    return {"plain": th, "whitened": thw, "random": rnd}, scale, proj_stats


def run(model_name, args):
    tok, model = S.get_model(model_name, DEV, torch.float16)
    dtype = next(model.parameters()).dtype
    out = {}

    for src in args.sources:
        ck = os.path.join(ART, "ckpt", f"{model_name.split('/')[-1]}__{src}.json")
        if not os.path.exists(ck):
            print(f"  !! no checkpoint for {src}; run snr_sweep.py first"); continue
        layer = json.load(open(ck))["best_layer"]
        print(f"\n  [{model_name.split('/')[-1]}] steering with {src} @ layer {layer}", flush=True)
        dirs, scale, pstats = fit_directions(model, tok, src, layer, args.cap, args.seed)
        print(f"    alpha unit = proj std {scale:.2f} "
              f"(act norm {pstats['mean_act_norm']:.1f}, class sep {pstats['class_sep']:.2f})",
              flush=True)

        for tgt in args.targets:
            pairs = load_pairs(tgt, args.pairs, args.seed)
            print(f"    -> scoring {tgt} ({len(pairs)} contrastive pairs)", flush=True)
            with Steerer(model, layer) as st:
                st.set(None, 0, DEV, dtype)
                base = score_pairs(model, tok, pairs, args.bs)
                rows = []
                for kind in ("plain", "whitened", "random"):
                    for a in args.alphas:
                        st.set(dirs[kind], a * scale, DEV, dtype)
                        sc = score_pairs(model, tok, pairs, args.bs)
                        shift = float(np.mean(sc - base))
                        flip = float(np.mean((sc > 0) != (base > 0)))
                        rows.append({"kind": kind, "alpha": a,
                                     "mean_shift": shift, "flip_rate": flip,
                                     "mean_score": float(sc.mean())})
                        print(f"       {kind:<9} a={a:+.1f}  shift={shift:+7.3f}  "
                              f"flip={flip:.3f}", flush=True)
                st.set(None, 0, DEV, dtype)
            out[f"{src}->{tgt}"] = {"layer": layer, "baseline": float(base.mean()),
                                    "n_pairs": len(pairs), "rows": rows}
        del dirs; gc.collect()

    del model, tok; gc.collect(); torch.mps.empty_cache()
    return out


# ---- cli -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--sources", default="cities,counterfact_true_false",
                    help="datasets whose truth direction we steer with")
    ap.add_argument("--targets", default="cities,counterfact_true_false",
                    help="datasets whose completions we score")
    ap.add_argument("--alphas", default="-2,-1,-0.5,0.5,1,2")
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ART, "steering.json"))
    args = ap.parse_args()
    args.sources = [s for s in args.sources.split(",") if s]
    args.targets = [s for s in args.targets.split(",") if s]
    args.alphas = [float(a) for a in args.alphas.split(",")]

    results = {}
    for m in [x for x in args.models.split(",") if x]:
        print(f"\n=== {m} ===", flush=True)
        try:
            results[m] = run(m, args)
        except Exception as e:
            print(f"  !! {m} failed: {type(e).__name__}: {e}", flush=True)
            results[m] = {"error": f"{type(e).__name__}: {e}"}
        with open(args.out, "w") as f:
            json.dump({"alphas": args.alphas, "models": results}, f, indent=2)
        print(f"  -> wrote {args.out}", flush=True)
    print("\ndone.")


if __name__ == "__main__":
    main()
