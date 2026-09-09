"""
Rogue-dimension observables on the main-tier datasets the post does not carry
through the steering analysis.

The twelve-dataset appendix infers that companies_true_false is "a second
instance of the rogue-dimension pattern" from its plain/whitened gap alone.
This measures the spectrum directly, at the three depths the post reasons
about, on the datasets whose selected layer sits deep and whose plain d' is
small. Same observables() as geometry_observables.py, so the numbers are
comparable to the cities / counterfact entries there.

Drafted with the assistance of Claude (Anthropic).
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("s", REPO / "scripts" / "snr_sweep.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
gs = importlib.util.spec_from_file_location("g", REPO / "scripts" / "geometry_observables.py")
G = importlib.util.module_from_spec(gs); gs.loader.exec_module(G)

MODEL = "EleutherAI/pythia-2.8b"
DATASETS = ["companies_true_false", "common_claim_true_false", "cities_cities_conj"]
LAYERS = [24, 28, 31]
DEV = os.environ.get("COVER_DEV", "mps")
OUT = REPO / "artifacts" / "geometry_extra_datasets.json"


def main():
    tok, model = S.get_model(MODEL, DEV, torch.float16)
    out = {}
    for ds in DATASETS:
        st, y = S.load_dataset(ds, cap=1199, seed=0)
        A = S.extract_all_layers(st, tok, model, DEV, 16)
        out[ds] = {}
        for L in LAYERS:
            o = G.observables(A[L].astype(np.float64), y, True); o.pop("theta")
            out[ds][str(L)] = o
            print(f"{ds:26s} L{L} PR={o['PR']:6.2f} lam1/tr={o['lambda1_over_trace']:.3f} "
                  f"lam1/lam2={o['lambda1_over_lambda2']:8.1f} cos(th,v1)={o['cos_theta_v1']:.3f} "
                  f"d_mm={o['d_mass_mean']:.2f}", flush=True)
        del A
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    main()
