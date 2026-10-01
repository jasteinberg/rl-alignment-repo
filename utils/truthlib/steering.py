"""
Steering harness (torch): the residual-stream hook, the behavioral score, and the
bookkeeping shared by the steering sweep, the extended null and the gradient run.

Moved verbatim from steer_completions.py (Steerer, score_pairs),
steer_confirm2.py (paths, activation cache, antisym, pair selection, SEED_PLAN)
and score_gradient.py (grad_for_texts, LOSS_SCALE); only module prefixes changed.
load_seed_cells / load_null_cell replace the per-script checkpoint readers.

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import glob
import json
import os

import numpy as np
import torch

from . import acts, data
from .estimators import auroc, chi_origin, class_gap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "artifacts")
CKPT = os.path.join(ART, "steer_ckpt")  # class-gap alpha units
ACTS = os.path.join(ART, "act_cache")
DEV = "mps"
LOSS_SCALE = 1024.0          # fp16 backward underflows without this

SEED_PLAN = {0.25: 10, 0.375: 10, 0.5: 10, 0.625: 5, 0.75: 5, 0.875: 10}


def tag(m): return m.split("/")[-1]


def cell_path(model, ds, layer, seed):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__s{seed}.json")


def null_path(model, ds, layer):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__NULL.json")


def act_path(model, ds):
    return os.path.join(ACTS, f"{tag(model)}__{ds}.npz")


def read_ckpt() -> str:
    """Directory the analyses read cells from: $STEER_CKPT if set, else CKPT.
    Writers always use CKPT."""
    return str(os.environ.get("STEER_CKPT", CKPT))


def load_seed_cells(model: str, ds: str, layer: int, ckpt: str | None = None) -> list[dict]:
    """Every seed's cell at one (model, dataset, layer), in sorted filename order
    (s0, s1, s10, s2, ...), the order the figures and tables were built with."""
    pat = os.path.join(ckpt or read_ckpt(), f"{tag(model)}__{ds}__L{layer}__s*.json")
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]


def load_null_cell(model: str, ds: str, layer: int, ckpt: str | None = None) -> dict | None:
    """The extended-null cell at one (model, dataset, layer), or None if not run."""
    p = os.path.join(ckpt or read_ckpt(), f"{tag(model)}__{ds}__L{layer}__NULL.json")
    return json.load(open(p)) if os.path.exists(p) else None


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
    from . import estimators as est
    tr, te = est.split_indices(y, seed=seed)
    th = est.mass_mean(X[tr], y[tr])
    if est.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    try:
        thw = est.fisher(X[tr], y[tr])
        if est.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    except Exception:
        thw = th.copy()
    return th, thw, tr, te


# ---- per-pair score gradient (moved from score_gradient) -------------------------
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


# ---- steering cells and their summary (moved from steer_confirm2, chi_whitening_analysis) ----
def run_cell(model, tok, Xl, y, pool, layer, alphas, seed, bs, dtype, n_pairs):
    """One steering cell: fit theta and theta_F on the seed's train half, then the
    odd and even response of the behavioral score at each alpha, in class-gap units."""
    X = Xl.astype(np.float64)
    th, thw, tr, te = fit_dirs(X, y, seed)
    pairs = pairs_for_seed(pool, seed, n_pairs)
    scale = class_gap(X[tr], y[tr])          # fit the unit on train, like theta
    rec = {"layer": layer, "seed": seed, "alpha_unit": scale,
           "alpha_unit_kind": "class_gap_norm",
           "sigma_along_theta": float(np.std(X @ th)),
           "d_prime_theta": scale / float(np.std(X @ th)),
           "n_pairs": len(pairs),
           "probe_auroc": auroc(X[te] @ th, y[te]),
           "probe_auroc_whitened": auroc(X[te] @ thw, y[te]), "alphas": {}}
    with Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = score_pairs(model, tok, pairs, bs)
        for a in alphas:
            ap, sp_ = antisym(model, tok, st, th, a, scale, pairs, base, bs, dtype)
            aw, sw_ = antisym(model, tok, st, thw, a, scale, pairs, base, bs, dtype)
            rec["alphas"][str(a)] = {"plain": {"antisym": ap, "sym": sp_},
                                     "whitened": {"antisym": aw, "sym": sw_}}
        st.set(None, 0, DEV, dtype)
    rec["baseline"] = float(base.mean())
    return rec, scale


def run_null(model, tok, Xl, y, pool, layer, alphas, bs, dtype, n_pairs, n_rand):
    """Steering null: the odd response along n_rand random unit directions, same unit."""
    X = Xl.astype(np.float64)
    scale = class_gap(X, y)                  # same unit as the theta arms
    pairs = pairs_for_seed(pool, 0, n_pairs)
    rng = np.random.default_rng(9000 + layer * 17)
    out = {"layer": layer, "n_rand": n_rand, "alphas": {}}
    with Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = score_pairs(model, tok, pairs, bs)
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


def chi_summary(model, dataset, layer, alphas):
    """chi (median, IQR) for both arms over seeds, and A(1) against the null:
    z-score and signed rank p."""
    seeds = load_seed_cells(model, dataset, layer)
    null = load_null_cell(model, dataset, layer)
    if not seeds or null is None:
        return None
    plain_chi, whit_chi, plain_a1, whit_a1 = [], [], [], []
    for s in seeds:
        al = s["alphas"]
        pv = [al[str(a)]["plain"]["antisym"] for a in alphas]
        wv = [al[str(a)]["whitened"]["antisym"] for a in alphas]
        plain_chi.append(chi_origin(alphas, pv))
        whit_chi.append(chi_origin(alphas, wv))
        plain_a1.append(al["1.0"]["plain"]["antisym"])
        whit_a1.append(al["1.0"]["whitened"]["antisym"])
    null1 = null["alphas"]["1.0"]
    null_draws = np.array(null1["draws"])

    def rank_p(obs):
        if obs >= 0:
            return float(np.mean(null_draws >= obs))
        return float(np.mean(null_draws <= obs))

    plain_mean, whit_mean = np.mean(plain_a1), np.mean(whit_a1)
    return dict(
        layer=layer, n=len(seeds),
        chi_plain=np.median(plain_chi), chi_plain_iqr=np.subtract(*np.percentile(plain_chi, [75, 25])),
        chi_whit=np.median(whit_chi), chi_whit_iqr=np.subtract(*np.percentile(whit_chi, [75, 25])),
        plain_pos_frac=float(np.mean(np.array(plain_a1) > 0)),
        whit_pos_frac=float(np.mean(np.array(whit_a1) > 0)),
        plain_z=(plain_mean - null1["antisym_mean"]) / null1["antisym_std"],
        whit_z=(whit_mean - null1["antisym_mean"]) / null1["antisym_std"],
        plain_p=rank_p(plain_mean), whit_p=rank_p(whit_mean),
        null_p95=null1["antisym_p95"], null_std=null1["antisym_std"],
    )
