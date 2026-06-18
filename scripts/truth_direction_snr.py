"""
scripts/truth_direction_snr.py
    python scripts/truth_direction_snr.py --model gpt2 --dataset cities [--whiten]
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env import ENV
from utils.train import get_device
from utils.probing import (load_lm, layer_sweep, residual_activations,
                           mass_mean_direction, direction_snr, project_out)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geometry_of_truth")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="cities")
    ap.add_argument("--whiten", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(f"{DATA}/{a.dataset}.csv")          # cols: statement, label
    statements, labels = df["statement"].tolist(), df["label"].to_numpy().astype(int)
    device = get_device(); print(f"device={device} model={a.model} n={len(statements)}")
    model, tok = load_lm(a.model, device=device)

    res = layer_sweep(model, tok, statements, labels, device=device, whiten=a.whiten)
    best = int(res["layer"][np.argmax(res["snr_dprime2"])])
    print(f"best layer {best}  d'^2={res['snr_dprime2'].max():.3f}  acc={res['accuracy'][best]:.3f}")

    X = residual_activations(model, tok, statements, best, device)
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    ks = [0, 1, 2, 4, 8, 16, 32]
    snr_proj = []
    for k in ks:
        Xp = project_out(X, Vt[:k]) if k else X
        th, _ = mass_mean_direction(Xp, labels)
        snr_proj.append(direction_snr(Xp, labels, th)["snr_dprime2"])

    os.makedirs(f"{ENV.REPO}/artifacts", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(res["layer"], res["snr_dprime2"], "-o"); ax[0].set(xlabel="layer", ylabel="SNR d'^2", title=f"{a.dataset}: truth-direction SNR")
    ax[1].plot(ks, snr_proj, "-o"); ax[1].set(xlabel="# top PCA dirs removed", ylabel="SNR d'^2", title=f"superposition probe (layer {best})")
    fig.tight_layout()
    p = f"{ENV.REPO}/artifacts/truth_snr_{a.model}_{a.dataset}.png"
    fig.savefig(p, dpi=140); print("saved", p)

if __name__ == "__main__":
    main()
