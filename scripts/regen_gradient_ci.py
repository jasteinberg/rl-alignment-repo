"""
Regenerate the bootstrap intervals on the score-gradient overlaps, with a FIXED
seed and a resample count large enough that the Monte-Carlo error of the
bootstrap sits below the precision quoted in the post.

Why this script exists: the point estimates cos(g, w) are deterministic
functions of the stored per-pair gradients, but the intervals were previously
drawn from an unseeded generator, so every regeneration of the JSON moved the
endpoints by 1-2 in the third decimal -- the precision the post quotes at. The
seed and B below are now part of the recorded method.

Reads artifacts/score_gradient_{name}_vectors.npz, rewrites the interval
triples [point, lo, hi] in artifacts/score_gradient_{name}.json in place,
leaving every other field untouched.

Drafted with the assistance of Claude (Anthropic).
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"

SEED = 0
B = 10000
NAMES = ["cities", "counterfact"]
KEYS = {"cos_g_v1": "v1", "cos_g_theta": "theta",
        "cos_g_theta_whitened": "theta_whitened", "cos_g_e2": "e2"}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def intervals(G, w, rng):
    """[point, lo, hi] for cos(mean_i g_i, w), bootstrapping over pairs."""
    n = len(G)
    point = cos(G.mean(0), w)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.array([cos(G[i].mean(0), w) for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [point, float(lo), float(hi)]


def main():
    for name in NAMES:
        z = np.load(ART / f"score_gradient_{name}_vectors.npz")
        path = ART / f"score_gradient_{name}.json"
        doc = json.load(open(path))
        layers = sorted(doc["layers"], key=int)
        for L in layers:
            G = z[f"L{L}_g_per_pair"].astype(np.float64)
            rng = np.random.default_rng(SEED + int(L))
            for jk, vk in KEYS.items():
                w = z[f"L{L}_{vk}"].astype(np.float64)
                old = doc["layers"][L][jk]
                new = intervals(G, w, rng)
                doc["layers"][L][jk] = new
                if jk == "cos_g_v1" or jk == "cos_g_theta_whitened":
                    print(f"  {name} L{L} {jk:<22} "
                          f"[{old[0]:+.4f} {old[1]:+.4f} {old[2]:+.4f}] -> "
                          f"[{new[0]:+.4f} {new[1]:+.4f} {new[2]:+.4f}]", flush=True)
        doc["note"] = (f"regenerated from *_vectors.npz; 95% bootstrap CIs over "
                       f"pairs, B={B}, seed={SEED}+layer")
        json.dump(doc, open(path, "w"), indent=2)
        print(f"-> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
