"""
Is whitening's perfect separation on `larger_than` a real linear feature, or an
artifact of inverting an ill-conditioned within-class covariance estimated from
~600 points in 2560 dimensions?

Controls:
  1. Shrinkage sweep: explicit shrinkage alpha in Sigma_a = (1-a)S + a*tr(S)/d*I.
     An artifact of the inverse should die as a -> 1 (Sigma -> isotropic, Fisher
     -> mass-mean). A real low-variance signal direction should persist to
     moderate a.
  2. Split-half covariance: estimate Sigma on one half, the mean difference on
     the other. Removes the coupling that lets Sigma^{-1} amplify the very noise
     directions that defined mu_1 - mu_0.
  3. Shuffled-label whitening: fit the Fisher direction to RANDOM labels. If it
     also separates its own held-out shuffled labels, the machinery is
     manufacturing separation from dimensional slack.
  4. Layerwise: report layer 1 alongside a late layer.

Drafted with the assistance of Claude (Anthropic).
"""
import sys, os, importlib.util
import numpy as np, pandas as pd, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("s", os.path.join(REPO, "scripts/snr_sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

from utils.env import ENV

D = ENV.GOT
MODEL, dev = "EleutherAI/pythia-2.8b", "mps"
LAYERS = [1, 28]

df = pd.read_csv(f"{D}/larger_than.csv").dropna(subset=["statement", "label"])
df["label"] = df["label"].astype(int)
tok, model = S.get_model(MODEL, dev, torch.float16)
A = S.extract_all_layers(df["statement"].tolist(), tok, model, dev, 16)
y = df["label"].to_numpy()


def fisher(Xtr, ytr, alpha):
    """Fisher direction with explicit shrinkage alpha toward isotropic."""
    Xc = Xtr.copy()
    for lab in (0, 1):
        Xc[ytr == lab] -= Xtr[ytr == lab].mean(0)
    Sg = np.cov(Xc, rowvar=False)
    d = Sg.shape[0]
    Sg = (1 - alpha) * Sg + alpha * (np.trace(Sg) / d) * np.eye(d)
    dmu = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    th = np.linalg.solve(Sg, dmu)
    return th / np.linalg.norm(th)


for L in LAYERS:
    X = A[L].astype(np.float64)
    tr, te = S.split_indices(y, seed=0)
    print(f"\n===== layer {L}  (N_train={len(tr)}, d={X.shape[1]}) =====")

    print("  -- 1. shrinkage sweep (alpha=1 is pure mass-mean) --")
    for a in (0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0):
        th = fisher(X[tr], y[tr], a)
        if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
        r = S.evaluate_direction(th, X[te], y[te])
        print(f"     alpha={a:<5} held-out d'={r['d_prime']:6.2f}  AUROC={r['auroc']:.4f}")

    print("  -- 2. split-half: Sigma and mu from disjoint halves --")
    a_, b_ = S.split_indices(y[tr], seed=7)
    ia, ib = tr[a_], tr[b_]
    Xc = X[ia].copy()
    for lab in (0, 1): Xc[y[ia] == lab] -= X[ia][y[ia] == lab].mean(0)
    from sklearn.covariance import LedoitWolf
    P = LedoitWolf(assume_centered=True).fit(Xc).precision_
    dmu = X[ib][y[ib] == 1].mean(0) - X[ib][y[ib] == 0].mean(0)
    th = P @ dmu; th /= np.linalg.norm(th)
    if S.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    r = S.evaluate_direction(th, X[te], y[te])
    print(f"     held-out d'={r['d_prime']:.2f}  AUROC={r['auroc']:.4f}")

    print("  -- 3. whitening fit to SHUFFLED labels --")
    rng = np.random.default_rng(0)
    ysh = rng.permutation(y)
    trs, tes = S.split_indices(ysh, seed=0)
    thw = S.whitened_direction(X[trs], ysh[trs])
    if S.auroc(X[trs] @ thw, ysh[trs]) < 0.5: thw = -thw
    r = S.evaluate_direction(thw, X[tes], ysh[tes])
    print(f"     held-out d'={r['d_prime']:.2f}  AUROC={r['auroc']:.4f}   (chance => 0.5)")
