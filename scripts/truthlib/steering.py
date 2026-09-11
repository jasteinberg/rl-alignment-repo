"""
Steering harness (torch): the residual-stream hook, the behavioral score, and the
bookkeeping shared by the steering sweep, the extended null and the gradient run.

Moved verbatim from steer_completions.py (Steerer, score_pairs) and
steer_confirm2.py (paths, activation cache, antisym, pair selection, SEED_PLAN);
only module prefixes changed.

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import glob
import json
import os

import numpy as np
import torch

from truthlib import acts, data

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    stmts, y = data.load_dataset(ds, cap=cap, seed=seed)
    print(f"    extracting {len(stmts)} activations...", flush=True)
    A = acts.extract_all_layers(stmts, tok, model, DEV, 16)
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


def pairs_for_seed(pool, seed, n):
    """Stable given the seed: seed k always selects the same evaluation subset."""
    rng = np.random.default_rng(777 + seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


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


def antisym(model, tok, st, vec, alpha, scale, pairs, base, bs, dtype):
    st.set(vec, +alpha * scale, DEV, dtype)
    sp = float(np.mean(score_pairs(model, tok, pairs, bs) - base))
    st.set(vec, -alpha * scale, DEV, dtype)
    sm = float(np.mean(score_pairs(model, tok, pairs, bs) - base))
    return 0.5 * (sp - sm), 0.5 * (sp + sm)


def fit_dirs(X, y, seed):
    """Plain and whitened directions on the seed's training half, each oriented so
    its training AUROC is at least 1/2 (moved from steer_confirm2.py)."""
    from truthlib import estimators as est
    tr, te = est.split_indices(y, seed=seed)
    th = est.mass_mean(X[tr], y[tr])
    if est.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    try:
        thw = est.fisher(X[tr], y[tr])
        if est.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    except Exception:
        thw = th.copy()
    return th, thw, tr, te
