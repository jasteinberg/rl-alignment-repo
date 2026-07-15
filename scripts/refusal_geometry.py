"""
scripts/refusal_geometry.py

GEOMETRY of the refusal representation in a safety-tuned model, compared with the
truth-direction geometry from the main sweep.

Research question (geometric, no intervention): does the refusal direction
r_hat = mean(harmful activations) - mean(harmless activations) align with the
model's dominant "rogue" / massive-activation direction v_1, the way the
counterfact truth direction does (cos = 0.994, decodes at chance)? Or is it a
genuine low-rogue-overlap feature?

This is a measurement of representation geometry. Prompts are used only as inputs
whose last-token activations are recorded; the model generates nothing here, and
no behaviour is modified. The observables are exactly those used elsewhere in the
post -- cos(r_hat, v_1), the covariance spectrum, participation ratio, and the
mass-mean vs Mahalanobis separation -- so refusal and truth are measured on the
same footing.

The causal validation of the refusal direction (that ablating it reduces refusal
and adding it induces refusal) is due to Arditi et al. (2024), "Refusal in
Language Models Is Mediated by a Single Direction", and is reproduced with credit
in scripts/refusal_causal_reproduction.py as a positive control. This file does
not intervene on the model at all.

Model: Qwen/Qwen1.5-1.8B-Chat (the small model studied by Arditi et al.).
Data:  harmful = AdvBench, harmless = Alpaca -- the standard contrast pair for
       locating the refusal direction.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("g", os.path.join(REPO, "scripts/geometry_observables.py"))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

ART, DEV = os.path.join(REPO, "artifacts"), "mps"
MODEL = "Qwen/Qwen1.5-1.8B-Chat"


def load_prompts(n=256, seed=0):
    """harmful = AdvBench instructions, harmless = Alpaca instructions.

    Used only to locate the refusal direction as a difference in means; these are
    inputs whose activations we record, not requests we fulfil.
    """
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    adv = load_dataset("walledai/AdvBench", split="train")["prompt"]
    alp = [x for x in load_dataset("tatsu-lab/alpaca", split="train")["instruction"] if x.strip()]
    h = list(rng.choice(adv, size=min(n // 2, len(adv)), replace=False))
    g = list(rng.choice(alp, size=min(n // 2, len(alp)), replace=False))
    return h, g


@torch.no_grad()
def last_token_acts(model, tok, prompts, layers, bs=8):
    outs = {L: [] for L in layers}
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        b = torch.arange(last.shape[0], device=DEV)
        for L in layers:
            outs[L].append(hs[L][b, last].float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in outs.items()}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    N = model.config.num_hidden_layers
    layers = sorted({int(round(f * N)) for f in (0.25, 0.36, 0.5, 0.625, 0.75, 0.875)})

    harmful, harmless = load_prompts()
    print(f"{len(harmful)} harmful / {len(harmless)} harmless prompts; layers {layers}", flush=True)
    Ah = last_token_acts(model, tok, harmful, layers)
    Ag = last_token_acts(model, tok, harmless, layers)

    # Build a two-class problem: y=1 harmful, y=0 harmless. The "direction" is the
    # difference in means, exactly the mass-mean estimator used for truth.
    out = {"model": MODEL, "layers": {}}
    print(f"\n{'L':>4} {'PR':>8} {'cos(r,v1)':>10} {'d_mm':>7} {'d_maha':>8} {'hidden':>7}", flush=True)
    for L in layers:
        X = np.vstack([Ah[L], Ag[L]]).astype(np.float64)
        y = np.concatenate([np.ones(len(Ah[L])), np.zeros(len(Ag[L]))]).astype(int)
        o = G.observables(X, y, shrink=True)
        o.pop("theta", None)
        out["layers"][str(L)] = o
        print(f"{L:>4} {o['PR']:>8.2f} {o['cos_theta_v1']:>10.3f} "
              f"{o['d_mass_mean']:>7.3f} {o['d_mahalanobis']:>8.2f} "
              f"{o['hidden_signal_ratio']:>7.1f}", flush=True)

    with open(os.path.join(ART, "refusal_geometry.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote artifacts/refusal_geometry.json")


if __name__ == "__main__":
    main()
