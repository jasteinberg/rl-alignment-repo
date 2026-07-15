#!/usr/bin/env python
"""
Estimator-vs-function-class analysis for counterfact steering, via
chi = dA/dalpha.
Reads existing artifacts/steer_ckpt/*.json -- NO new activations/compute.

Question: is weak/wrong-signed counterfact steering explained by
(a) rogue-dimension collapse of the mass-mean estimator (our diagnosis), or
(b) the linear function class itself being too weak to move the behaviour?

Note on (b): this is a natural hypothesis to rule out, not a position taken from
the literature. Braun et al. (2025), "Understanding (Un)Reliability of Steering
Vectors in Language Models" (arXiv:2505.22637), argue something adjacent but
different -- that steerability tracks the directional coherence of a behaviour's
activation differences and their separability along the difference-of-means line,
i.e. steering fails when the behaviour is not a coherent linear direction. The
result here supplies a mechanism for one such failure rather than contradicting it.

Test: if whitening (Mahalanobis-correcting for the rogue dimension) alone
recovers correctly-signed, non-null causal steering with a single rank-1
direction, that supports (a) -- no need to invoke subspace/nonlinear
expressivity limits. If whitened steering stays weak/wrong-signed too,
that's evidence for (b) or a mix.
"""
import json
import os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("STEER_CKPT", REPO / "artifacts" / "steer_ckpt"))
ALPHAS = [0.5, 1.0, 2.0, 4.0]
MODEL = "pythia-2.8b"
DATASETS = ["cities", "counterfact_true_false"]
LAYERS = {"cities": [8, 12, 16, 20, 24, 28],
          "counterfact_true_false": [8, 12, 16, 20, 24, 28]}


def load(model, dataset, layer):
    seeds = []
    for f in sorted(CKPT.glob(f"{model}__{dataset}__L{layer}__s*.json")):
        seeds.append(json.load(open(f)))
    null_f = CKPT / f"{model}__{dataset}__L{layer}__NULL.json"
    null = json.load(open(null_f)) if null_f.exists() else None
    return seeds, null


def chi_origin(alphas, values):
    """Susceptibility via least-squares fit forced through origin."""
    a = np.array(alphas, dtype=float)
    v = np.array(values, dtype=float)
    return float((a * v).sum() / (a * a).sum())


def analyze_layer(model, dataset, layer):
    seeds, null = load(model, dataset, layer)
    if not seeds or null is None:
        return None
    plain_chi, whit_chi, plain_a1, whit_a1 = [], [], [], []
    for s in seeds:
        al = s["alphas"]
        pv = [al[str(a)]["plain"]["antisym"] for a in ALPHAS]
        wv = [al[str(a)]["whitened"]["antisym"] for a in ALPHAS]
        plain_chi.append(chi_origin(ALPHAS, pv))
        whit_chi.append(chi_origin(ALPHAS, wv))
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


def main():
    for dataset in DATASETS:
        print(f"\n=== {MODEL} / {dataset} ===")
        hdr = f"{'L':>3} {'n':>2} | {'chi_plain':>10} {'chi_whit':>10} | " \
              f"{'plain+%':>7} {'whit+%':>7} | {'plain_z':>7} {'whit_z':>7} | " \
              f"{'plain_p':>7} {'whit_p':>7}"
        print(hdr)
        print("-" * len(hdr))
        for L in LAYERS[dataset]:
            r = analyze_layer(MODEL, dataset, L)
            if r is None:
                print(f"{L:>3}  -- missing --")
                continue
            print(f"{r['layer']:>3} {r['n']:>2} | "
                  f"{r['chi_plain']:+10.4f} {r['chi_whit']:+10.4f} | "
                  f"{r['plain_pos_frac']*100:6.0f}% {r['whit_pos_frac']*100:6.0f}% | "
                  f"{r['plain_z']:+7.2f} {r['whit_z']:+7.2f} | "
                  f"{r['plain_p']:7.3f} {r['whit_p']:7.3f}")


if __name__ == "__main__":
    main()
