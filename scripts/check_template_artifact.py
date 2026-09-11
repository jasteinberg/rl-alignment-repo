"""
Frame-disjoint control: is whitening's power on `larger_than` real, or is it
decoding template noise that transfers because train/test are near-copies?

`larger_than` has ONE frame ("X is larger than Y", every statement 5 words),
so a random split puts near-duplicate sentences on both sides. Here we split by
the FIRST numeral: no number that appears in a training statement appears in a
test statement. A real truth/comparison feature should survive; a direction that
exploits an ill-conditioned Sigma^{-1} on shared template noise should collapse.

Drafted with the assistance of Claude (Anthropic).
"""
import sys, json, re
import numpy as np, pandas as pd, torch, os
from truthlib import acts
from truthlib import estimators as est

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.env import ENV

D = ENV.GOT
MODEL, LAYER = "EleutherAI/pythia-2.8b", 28
dev = "mps"

df = pd.read_csv(f"{D}/larger_than.csv").dropna(subset=["statement", "label"])
df["label"] = df["label"].astype(int)
first = df["statement"].str.split().str[0].str.lower()
df["frame"] = first

tok, model = acts.get_model(MODEL, dev, torch.float16)
A = acts.extract_all_layers(df["statement"].tolist(), tok, model, dev, 16)
X = A[LAYER].astype(np.float64); y = df["label"].to_numpy()
print(f"N={len(y)}  d={X.shape[1]}  layer={LAYER}")

def report(tag, tr, te):
    th = est.mass_mean(X[tr], y[tr])
    if est.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    p = est.evaluate_direction(th, X[te], y[te])
    thw = est.fisher(X[tr], y[tr])
    if est.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    w = est.evaluate_direction(thw, X[te], y[te])
    print(f"{tag:<22} plain d'={p['d_prime']:.2f} AUC={p['auroc']:.3f}   "
          f"whitened d'={w['d_prime']:.2f} AUC={w['auroc']:.3f}")

# 1) random split (what the sweep does)
tr, te = est.split_indices(y, seed=0)
report("random split", tr, te)

# 2) frame-disjoint: first numeral disjoint between halves
rng = np.random.default_rng(0)
frames = df["frame"].unique(); rng.shuffle(frames)
half = set(frames[: len(frames)//2])
mask = df["frame"].isin(half).to_numpy()
tr2, te2 = np.where(mask)[0], np.where(~mask)[0]
print(f"frame-disjoint: {len(tr2)} train / {len(te2)} test, "
      f"{len(half)}/{len(frames)} first-numerals held out")
report("frame-disjoint split", tr2, te2)
