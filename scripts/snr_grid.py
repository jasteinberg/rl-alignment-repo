"""
scripts/snr_grid.py
Domain x scale x estimator grid for the truth-direction SNR.
Loads each model once; extracts residual activations once per (dataset, layer)
and computes both the plain and whitened (Fisher/LDA) mass-mean probe from them.
Writes artifacts/snr_grid.csv and per-condition SNR-vs-layer plots.
Scaffolded with assistance from Claude (Anthropic).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env import ENV
from utils.train import get_device
from utils.probing import (load_lm, residual_activations, mass_mean_direction,
                           direction_snr, probe_accuracy)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geometry_of_truth")
MODELS = ["gpt2", "EleutherAI/pythia-410m"]
DATASETS = ["cities", "larger_than", "sp_en_trans"]
CAP = 1000

device = get_device()
print("device", device, flush=True)
rows = []
for model_name in MODELS:
    print(f"loading {model_name} ...", flush=True)
    model, tok = load_lm(model_name, device=device)
    nL = model.config.num_hidden_layers
    for ds in DATASETS:
        df = pd.read_csv(f"{DATA}/{ds}.csv")
        statements = df["statement"].tolist()
        y = df["label"].to_numpy().astype(int)
        rng = np.random.default_rng(0)
        if len(y) > CAP:
            idx = rng.permutation(len(y))[:CAP]
            statements = [statements[i] for i in idx]; y = y[idx]
        perm = rng.permutation(len(y)); ntr = int(0.5 * len(y))
        tr, te = perm[:ntr], perm[ntr:]
        curves = {"plain": {"snr": [], "acc": []}, "whiten": {"snr": [], "acc": []}}
        for L in range(nL + 1):
            X = residual_activations(model, tok, statements, L, device)
            for est, wh in [("plain", False), ("whiten", True)]:
                th, b = mass_mean_direction(X[tr], y[tr], whiten=wh)
                curves[est]["snr"].append(direction_snr(X[te], y[te], th)["snr_dprime2"])
                curves[est]["acc"].append(probe_accuracy(X[te], y[te], th, b))
        for est in ("plain", "whiten"):
            snr = np.array(curves[est]["snr"]); acc = np.array(curves[est]["acc"])
            bl = int(np.argmax(snr))
            rows.append({"model": model_name, "dataset": ds, "estimator": est,
                         "best_layer": bl, "n_layers": nL, "rel_depth": round(bl / nL, 2),
                         "dprime2": float(snr[bl]), "acc": float(acc[bl]), "n": int(len(y))})
            print(f"{model_name:22s} {ds:13s} {est:6s} L={bl:2d}/{nL} "
                  f"d'^2={snr[bl]:.3f} acc={acc[bl]:.3f}", flush=True)
        mtag = model_name.replace("/", "_")
        fig, ax = plt.subplots(figsize=(6, 4))
        for est in ("plain", "whiten"):
            ax.plot(range(nL + 1), curves[est]["snr"], "-o", label=est)
        ax.set(xlabel="layer", ylabel="SNR d'^2", title=f"{mtag} / {ds}"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{ENV.REPO}/artifacts/grid_{mtag}_{ds}.png", dpi=140)
        plt.close(fig)

res = pd.DataFrame(rows)
res.to_csv(f"{ENV.REPO}/artifacts/snr_grid.csv", index=False)
print("\n=== GRID ===", flush=True)
print(res.to_string(index=False), flush=True)
print("saved", f"{ENV.REPO}/artifacts/snr_grid.csv", flush=True)
