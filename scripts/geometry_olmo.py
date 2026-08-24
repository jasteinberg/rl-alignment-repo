"""
Does the rogue-dimension signature replicate outside the Pythia family?

Runs the geometry observables (PR, lambda_1/trace, lambda_1/lambda_2,
|cos(theta_hat, v_1)|) on OLMo-2-1B for counterfact_true_false and cities, using
the same loaders and the same estimator as the Pythia results, so the numbers are
directly comparable. Activations only -- no steering, no GPU.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_sweep import load_dataset, get_model, extract_all_layers, cap_for, ART
from geometry_observables import observables

MODEL = os.environ.get("OLMO_MODEL", "allenai/OLMo-2-0425-1B")
DATASETS = ["counterfact_true_false", "cities"]
OUT = Path(os.environ.get("OUT", ART / "geometry_olmo.json"))
DEVICE = "mps"

tok, model = get_model(MODEL, DEVICE, torch.float32)
res = {"model": MODEL, "datasets": {}}

for ds in DATASETS:
    stmts, y = load_dataset(ds, cap=cap_for(ds))   # same cap as the pythia runs
    y = np.asarray(y)
    A = extract_all_layers(stmts, tok, model, DEVICE, batch_size=16)
    nL = A.shape[0]
    print(f"\n=== {MODEL} / {ds}  ({len(y)} rows, {nL} layers) ===")
    print(f"{'L':>4} {'PR':>8} {'l1/tr':>8} {'l1/l2':>9} {'cos_v1':>8} {'d_mm':>7} {'d_maha':>8}")
    rows = []
    for L in range(nL):
        X = np.asarray(A[L], dtype=np.float64)
        o = observables(X, y)
        o.pop("theta", None)      # 2048 floats per layer; not needed
        o["layer"] = L
        rows.append(o)
        print(f"{L:>4} {o['PR']:>8.2f} {o['lambda1_over_trace']:>8.3f} "
              f"{o['lambda1_over_lambda2']:>9.1f} {o['cos_theta_v1']:>8.3f} "
              f"{o['d_mass_mean']:>7.2f} {o['d_mahalanobis']:>8.2f}")
    res["datasets"][ds] = rows

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
