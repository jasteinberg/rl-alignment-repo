"""
scripts/refusal_causal_reproduction.py

APPENDIX / POSITIVE CONTROL: reproduce, with credit, the central finding of

    Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee & Nanda (2024),
    "Refusal in Language Models Is Mediated by a Single Direction",

namely that a single difference-in-means direction is causally responsible for a
safety-tuned model's refusal behaviour: removing the direction reduces refusal and
adding it induces refusal. We reproduce BOTH halves of that symmetric effect, which
is what distinguishes a faithful mechanistic reproduction from a one-sided attempt
to defeat a safeguard -- the point is to confirm a published circuit, not to
maximise any jailbreak.

Why this is in the post: it is a POSITIVE CONTROL. The counterfact "truth
direction" decodes at chance yet its mass-mean vector is essentially the rogue
dimension (cos(theta, v1) = 0.994); we need to show what a GENUINELY causal
difference-in-means direction looks like for contrast. Arditi et al.'s refusal
direction is the canonical verified example, so we borrow it as scaffolding and
credit it as such. The novel contribution of the post is the geometry
(scripts/refusal_geometry.py), not this reproduction.

Scope and safety: this measures the effect size of a KNOWN circuit on the standard
AdvBench/Alpaca benchmark used by Arditi et al., via a refusal-token probability
proxy on a small held-out set. It generates no harmful content, chains to no
downstream task, and reports the mechanism symmetrically (ablate AND induce). It is
a measurement of a documented safety mechanism, run to calibrate this post's null.

Model: Qwen/Qwen1.5-1.8B-Chat (the small model studied by Arditi et al.).
Data:  harmful = AdvBench, harmless = Alpaca.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, gc, argparse
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
DEV = "mps"
MODEL = "Qwen/Qwen1.5-1.8B-Chat"


REFUSAL_TOKS = ["I", "I'm", "As", "Sorry", "I cannot", "I can", "Unfortunately"]


def load_prompts(n=256, seed=0):
    """harmful = AdvBench, harmless = Alpaca; balanced, chat-formatted."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    adv = load_dataset("walledai/AdvBench", split="train")["prompt"]
    alp = [x for x in load_dataset("tatsu-lab/alpaca", split="train")["instruction"] if x.strip()]
    h = list(rng.choice(adv, size=min(n // 2, len(adv)), replace=False))
    g = list(rng.choice(alp, size=min(n // 2, len(alp)), replace=False))
    return h, g


def chat_wrap(tok, instr):
    msgs = [{"role": "user", "content": instr}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def last_token_acts(model, tok, prompts, layers, bs=8):
    """Last-token residual activations at the given layers."""
    outs = {L: [] for L in layers}
    for i in range(0, len(prompts), bs):
        texts = [chat_wrap(tok, p) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        b = torch.arange(last.shape[0], device=DEV)
        for L in layers:
            outs[L].append(hs[L][b, last].float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in outs.items()}


@torch.no_grad()
def refusal_score(model, tok, prompts, refusal_ids, bs=8):
    """Proxy: P(first generated token is a refusal token), mean over prompts."""
    sc = []
    for i in range(0, len(prompts), bs):
        texts = [chat_wrap(tok, p) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        b = torch.arange(last.shape[0], device=DEV)
        p = logits[b, last].float().softmax(-1)
        sc.append(p[:, refusal_ids].sum(-1).cpu().numpy())
    return np.concatenate(sc)


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v


def spectrum_and_dirs(Xh, Xg):
    """Refusal dir r = mean(harmful) - mean(harmless); its overlap with v1 of the
    POOLED covariance (the analogue of within-class covariance for two groups)."""
    X = np.vstack([Xh, Xg])
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(C); w = w[::-1]; V = V[:, ::-1]
    v1 = V[:, 0]
    r = unit(Xh.mean(0) - Xg.mean(0))
    if r @ v1 < 0: v1 = -v1
    sp = {"lambda1_over_trace": float(w[0] / w.sum()),
          "lambda1": float(w[0]), "lambda2": float(w[1]),
          "cos_r_v1": float(abs(r @ v1)),
          "var_along_r_frac": float((r @ C @ r) / w.sum()),
          "delta_over_sqrt_trace": float(np.linalg.norm(Xh.mean(0) - Xg.mean(0)) / np.sqrt(w.sum()))}
    # r with v1 projected out
    Xh_p = Xh - np.outer(Xh @ v1, v1); Xg_p = Xg - np.outer(Xg @ v1, v1)
    r_perp = unit(Xh_p.mean(0) - Xg_p.mean(0))
    return sp, {"r": r, "v1": v1, "r_perp": r_perp}, C


class Steer:
    def __init__(self, model, layer, mode):
        self.block = model.model.layers[layer]; self.mode = mode
        self.vec = None; self.h = None
    def __enter__(self):
        def hook(m, i, o):
            if self.vec is None: return o
            hs = o[0] if isinstance(o, tuple) else o
            if self.mode == "add":
                hs = hs + self.vec.to(hs.dtype)
            else:  # ablate: remove the component along the unit vec
                u = self.vec.to(hs.dtype)
                hs = hs - (hs @ u).unsqueeze(-1) * u
            return (hs,) + o[1:] if isinstance(o, tuple) else hs
        self.h = self.block.register_forward_hook(hook); return self
    def __exit__(self, *a):
        if self.h: self.h.remove()
    def set_add(self, v, a, dt): self.vec = None if v is None else a * torch.tensor(v, device=DEV, dtype=dt)
    def set_ablate(self, v, dt): self.vec = None if v is None else torch.tensor(v, device=DEV, dtype=dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ART, "refusal_rogue.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # generation-style: last token is the real one
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    dt = next(model.parameters()).dtype
    N = model.config.num_hidden_layers
    layers = sorted({int(round(f * N)) for f in (0.36, 0.5, 0.625, 0.75)})
    refusal_ids = sorted({tok(t).input_ids[0] for t in REFUSAL_TOKS})

    harmful, harmless = load_prompts(args.n)
    print(f"{len(harmful)} harmful / {len(harmless)} harmless; layers {layers}", flush=True)
    Ah = last_token_acts(model, tok, harmful, layers, args.bs)
    Ag = last_token_acts(model, tok, harmless, layers, args.bs)

    base_h = refusal_score(model, tok, harmful, refusal_ids, args.bs).mean()
    base_g = refusal_score(model, tok, harmless, refusal_ids, args.bs).mean()
    print(f"baseline refusal: harmful={base_h:.3f}  harmless={base_g:.3f}", flush=True)

    out = {"model": MODEL, "baseline": {"harmful": float(base_h), "harmless": float(base_g)},
           "layers": {}}
    for L in layers:
        sp, dirs, C = spectrum_and_dirs(Ah[L].astype(np.float64), Ag[L].astype(np.float64))
        print(f"\n--- layer {L} ---  lambda1/tr={sp['lambda1_over_trace']:.4f}  "
              f"|cos(r,v1)|={sp['cos_r_v1']:.3f}  var_along_r={sp['var_along_r_frac']:.3f}", flush=True)
        rec = {"spectrum": sp, "arms": {}}
        # norm-matched across arms: alpha * ||delta||, delta = mean_h - mean_g
        scale = float(np.linalg.norm(Ah[L].mean(0) - Ag[L].mean(0)))
        for name in ("r", "v1", "r_perp", "random"):
            if name == "random":
                rng = np.random.default_rng(7 + L); u = unit(rng.standard_normal(C.shape[0]))
            else:
                u = dirs[name]
            # ABLATE across all layers >= L : does refusal on harmful drop?
            with Steer(model, L, "ablate") as st:
                st.set_ablate(u, dt)
                abl_h = refusal_score(model, tok, harmful, refusal_ids, args.bs).mean()
            # ADD at layer L: does refusal on harmless rise?
            with Steer(model, L, "add") as st:
                st.set_add(u, 1.0 * scale, dt)
                add_g = refusal_score(model, tok, harmless, refusal_ids, args.bs).mean()
            rec["arms"][name] = {"ablate_harmful_refusal": float(abl_h),
                                 "add_harmless_refusal": float(add_g),
                                 "drop": float(base_h - abl_h), "induce": float(add_g - base_g)}
            print(f"    {name:<8} ablate->harmful refusal {abl_h:.3f} (drop {base_h-abl_h:+.3f})  "
                  f"add->harmless refusal {add_g:.3f} (induce {add_g-base_g:+.3f})", flush=True)
        out["layers"][str(L)] = rec
        with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}\ndone.")


if __name__ == "__main__":
    main()
