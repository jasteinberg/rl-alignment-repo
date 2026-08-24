"""scripts/transfer_to_likely.py

Closes the distractor test the post currently flags as not run.

Fitting on `likely` and scoring on `likely` says only that a probability axis
exists. The test Marks & Tegmark designed the set for is CROSS-dataset: take a
direction fitted on truth statements and evaluate it on `likely`. If a truth
direction is really reading textual plausibility, it should separate `likely`;
if it is reading truth, it should sit at chance there.

Runs both directions of the comparison (truth-fitted -> likely, and
likely-fitted -> each truth set), for the plain and whitened arms, at each
dataset's own selected layer from snr_sweep.json.

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snr_sweep as S

ART = Path(__file__).resolve().parent.parent / "artifacts"
DEV = "mps"


def fit(X, y, seed=0):
    tr, te = S.split_indices(y, seed=seed)
    th = S.mass_mean_direction(X[tr], y[tr])
    if S.auroc(X[tr] @ th, y[tr]) < 0.5:
        th = -th
    try:
        thw = S.whitened_direction(X[tr], y[tr])
        if S.auroc(X[tr] @ thw, y[tr]) < 0.5:
            thw = -thw
    except Exception:
        thw = th.copy()
    return th, thw, te


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", default=str(ART / "snr_sweep.json"))
    ap.add_argument("--out", default=str(ART / "transfer_likely.json"))
    args = ap.parse_args()

    sweep = json.load(open(args.sweep))
    m = sweep["models"][args.model]
    best = {ds: v["best_layer"] for ds, v in m["datasets"].items()}
    truth_sets = [d for d in m["transfer"].keys()]      # the 9 main-tier sets

    tok, model = S.get_model(args.model, DEV, torch.float16)

    # activations at every layer, per dataset, one pass each
    A, Y = {}, {}
    for ds in truth_sets + ["likely"]:
        stmts, y = S.load_dataset(ds, cap=args.cap, seed=args.seed)
        A[ds] = S.extract_all_layers(stmts, tok, model, DEV, 16)
        Y[ds] = y
        print(f"  extracted {ds} ({len(y)})", flush=True)

    out = {"model": args.model, "note": "AUROC of a direction fitted on `train_on` "
           "and evaluated on `eval_on`, at the layer selected for `train_on`",
           "rows": []}

    # truth-fitted -> likely, and the reverse, both arms
    for a, b in [(t, "likely") for t in truth_sets] + \
                [("likely", t) for t in truth_sets]:
        L = best[a]
        if L >= len(A[b]):
            continue
        Xa, Xb = A[a][L].astype(np.float64), A[b][L].astype(np.float64)
        th, thw, _ = fit(Xa, Y[a], args.seed)
        _, teb = S.split_indices(Y[b], seed=args.seed)
        row = {"train_on": a, "eval_on": b, "layer": int(L),
               "auroc_plain": float(S.auroc(Xb[teb] @ th, Y[b][teb])),
               "auroc_whitened": float(S.auroc(Xb[teb] @ thw, Y[b][teb]))}
        out["rows"].append(row)
        print(f"  {a:26s} -> {b:26s} L{L:2d} "
              f"plain={row['auroc_plain']:.3f} whit={row['auroc_whitened']:.3f}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        gc.collect()

    print(f"\nwrote {args.out}\ndone.")


if __name__ == "__main__":
    main()
