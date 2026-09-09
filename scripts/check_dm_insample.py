"""
Is the in-sample Mahalanobis separation d'_M = sqrt(delta' Sigma^-1 delta) a
capability ceiling, or is it mostly the same finite-sample slack that inflates
every fitted in-sample quantity at N << 2d?

The post quotes d'_M = 1.03 at counterfact layer 28 as "what the geometry is
capable of", computed on the full set with a Ledoit-Wolf precision. That is a
fitted, in-sample number at n = 1198 in d = 2560. Result (1) of the same post
says such numbers are inflated by noise alone. So: recompute it under shuffled
labels with the identical recipe. If the shuffled value is a large fraction of
1.03, the number is not a ceiling.

Also recorded: the same quantity on the training half alone (n = 599), to see
how the inflation moves with n, and the Ledoit-Wolf intensity each fit chose.

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import json
import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("s", REPO / "scripts" / "snr_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
gspec = importlib.util.spec_from_file_location("g", REPO / "scripts" / "geometry_observables.py")
G = importlib.util.module_from_spec(gspec)
gspec.loader.exec_module(G)

MODEL = "EleutherAI/pythia-2.8b"
DATASETS = ["counterfact_true_false", "cities"]
LAYERS = [24, 28, 32]
N_SHUF = 20
DEV = os.environ.get("COVER_DEV", "mps")
OUT = REPO / "artifacts" / "check_dm_insample.json"


def dm_insample(X, y):
    """Identical recipe to geometry_observables.observables, plus the LW intensity."""
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    Xc = G.within_class_cov(X, y)
    lw = LedoitWolf(assume_centered=True).fit(Xc)
    dm = float(np.sqrt(max(delta @ lw.precision_ @ delta, 0.0)))
    return dm, float(lw.shrinkage_)


def main():
    tok, model = S.get_model(MODEL, DEV, torch.float16)
    out = {"config": {"model": MODEL, "layers": LAYERS, "n_shuffles": N_SHUF}, "results": {}}
    for ds in DATASETS:
        stmts, y = S.load_dataset(ds, cap=1199, seed=0)
        A = S.extract_all_layers(stmts, tok, model, DEV, 16)
        res = {}
        rng = np.random.default_rng(0)
        for L in LAYERS:
            X = A[L].astype(np.float64)
            dm_true, rho_true = dm_insample(X, y)
            shuf = [dm_insample(X, rng.permutation(y)) for _ in range(N_SHUF)]
            dm_s = np.array([s[0] for s in shuf]); rho_s = np.array([s[1] for s in shuf])
            # training half only, true labels
            tr, te = S.split_indices(y, seed=0) if hasattr(S, "split_indices") else (None, None)
            if tr is not None:
                dm_half, rho_half = dm_insample(X[tr], y[tr])
                shuf_half = [dm_insample(X[tr], rng.permutation(y[tr]))[0] for _ in range(N_SHUF)]
            else:
                dm_half, rho_half, shuf_half = float("nan"), float("nan"), [float("nan")]
            res[str(L)] = {
                "n_full": int(len(y)),
                "dm_true_full": dm_true, "rho_true_full": rho_true,
                "dm_shuffled_full_mean": float(dm_s.mean()),
                "dm_shuffled_full_sd": float(dm_s.std(ddof=1)),
                "dm_shuffled_full_p95": float(np.percentile(dm_s, 95)),
                "rho_shuffled_full_mean": float(rho_s.mean()),
                "dm_true_half": dm_half, "rho_true_half": rho_half,
                "dm_shuffled_half_mean": float(np.mean(shuf_half)),
                "dm_shuffled_half_sd": float(np.std(shuf_half, ddof=1)) if len(shuf_half) > 1 else float("nan"),
            }
            r = res[str(L)]
            print(f"[{ds}] L{L}  full n={r['n_full']}: true d'_M={dm_true:.3f} (rho={rho_true:.3f})"
                  f"  shuffled={r['dm_shuffled_full_mean']:.3f}±{r['dm_shuffled_full_sd']:.3f}"
                  f" p95={r['dm_shuffled_full_p95']:.3f} (rho={r['rho_shuffled_full_mean']:.3f})"
                  f"  | half: true={dm_half:.3f} shuffled={r['dm_shuffled_half_mean']:.3f}",
                  flush=True)
        out["results"][ds] = res
        del A
        gc.collect()
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    main()
