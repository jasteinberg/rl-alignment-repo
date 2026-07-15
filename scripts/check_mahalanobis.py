"""
scripts/check_mahalanobis.py

Is the hidden_signal_ratio (d_Maha / d_mass-mean, up to 44x on 2.8b/cities L8)
real, or is it Sigma^{-1} amplifying estimation noise at N~1200 << d=2560?

The mass-mean probe sees d'=0.11 at 2.8b/cities layer 8; the Mahalanobis
(optimal-linear) separation is 4.94. Either the truth is genuinely present but
invisible to the naive estimator (real hidden signal), or the shrunk inverse is
manufacturing separation from dimensional slack. We already got burned once by a
Sigma^{-1} artifact (larger_than whitened AUROC = 1.000), so this needs controls.

Three controls, all evaluated HELD-OUT (fit on train, score on test):

  1. Shrinkage sweep. Vary alpha in Sigma_a = (1-a)S + a*tr(S)/d*I explicitly.
     A real hidden signal should plateau across a wide middle range of alpha; an
     artifact should spike near alpha=0 (raw cov, ill-conditioned) and collapse
     toward the mass-mean value as alpha->1 (Sigma -> isotropic).

  2. Split-half covariance. Estimate Sigma on one half of train, the mean
     difference on the other. Removes the coupling that lets Sigma^{-1} amplify
     the very noise directions that defined mu_1 - mu_0.

  3. Shuffled labels. Fit the whole Mahalanobis pipeline to a random permutation
     of the labels. If d_Maha is still large, the separation is coming from the
     ambient geometry, not from truth.

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, glob, importlib.util
import numpy as np
from sklearn.covariance import LedoitWolf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "check_mahalanobis.json")



def split(y, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]; rng.shuffle(idx)
        k = len(idx) // 2
        tr.append(idx[:k]); te.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)


def within(X, y):
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] -= X[y == lab].mean(0)
    return Xc


def d_prime_dir(theta, X, y):
    z = X @ theta; z1, z0 = z[y == 1], z[y == 0]
    den = 0.5 * (z1.var(ddof=1) + z0.var(ddof=1))
    return abs(z1.mean() - z0.mean()) / np.sqrt(den) if den > 0 else 0.0


def fisher(Xtr, ytr, alpha):
    """Fisher direction with explicit shrinkage alpha toward isotropic."""
    Xc = within(Xtr, ytr)
    Sg = np.cov(Xc, rowvar=False); d = Sg.shape[0]
    Sg = (1 - alpha) * Sg + alpha * (np.trace(Sg) / d) * np.eye(d)
    dmu = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    th = np.linalg.solve(Sg, dmu)
    return th / np.linalg.norm(th)


def maha_dprime(Xtr, ytr, Xte, yte, alpha=None, shrink_lw=False):
    """Held-out d' of the Fisher direction fit on train."""
    if shrink_lw:
        Xc = within(Xtr, ytr)
        P = LedoitWolf(assume_centered=True).fit(Xc).precision_
        dmu = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        th = P @ dmu; th /= np.linalg.norm(th)
    else:
        th = fisher(Xtr, ytr, alpha)
    if d_prime_dir(th, Xtr, ytr) < 0:
        th = -th
    return d_prime_dir(th, Xte, yte)


def analyze(X, y, layer):
    tr, te = split(y, seed=0)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    th_mm = (Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
    th_mm /= np.linalg.norm(th_mm)
    d_mm = d_prime_dir(th_mm, Xte, yte)

    rec = {"layer": layer, "d_mass_mean_heldout": float(d_mm), "shrinkage": {},
           "d_maha_lw_heldout": float(maha_dprime(Xtr, ytr, Xte, yte, shrink_lw=True))}

    # 1. shrinkage sweep
    for a in (0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0):
        try:
            rec["shrinkage"][str(a)] = float(maha_dprime(Xtr, ytr, Xte, yte, alpha=a))
        except np.linalg.LinAlgError:
            rec["shrinkage"][str(a)] = None

    # 2. split-half covariance (Sigma and delta from disjoint halves of train)
    a_, b_ = split(ytr, seed=7)
    Xc = within(Xtr[a_], ytr[a_])
    P = LedoitWolf(assume_centered=True).fit(Xc).precision_
    dmu = Xtr[b_][ytr[b_] == 1].mean(0) - Xtr[b_][ytr[b_] == 0].mean(0)
    th = P @ dmu; th /= np.linalg.norm(th)
    if d_prime_dir(th, Xtr, ytr) < 0: th = -th
    rec["d_maha_splithalf_heldout"] = float(d_prime_dir(th, Xte, yte))

    # 3. shuffled labels
    rng = np.random.default_rng(0)
    ysh = rng.permutation(y)
    trs, tes = split(ysh, seed=0)
    rec["d_maha_shuffled_heldout"] = float(
        maha_dprime(X[trs], ysh[trs], X[tes], ysh[tes], shrink_lw=True))
    return rec


def main():
    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        z = np.load(p); y = z["y"]
        layers = sorted([int(k[1:]) for k in z.files if k.startswith("L")])
        print(f"\n=== {key} ===", flush=True)
        print(f"{'L':>4} {'d_mm':>7} {'d_maha':>8} {'splithalf':>10} "
              f"{'SHUFFLED':>9} {'a=0':>7} {'a=.1':>7} {'a=.9':>7}", flush=True)
        rows = []
        for L in layers:
            r = analyze(z[f"L{L}"].astype(np.float64), y, L)
            rows.append(r)
            sh = r["shrinkage"]
            print(f"{L:>4} {r['d_mass_mean_heldout']:>7.3f} "
                  f"{r['d_maha_lw_heldout']:>8.3f} {r['d_maha_splithalf_heldout']:>10.3f} "
                  f"{r['d_maha_shuffled_heldout']:>9.3f} "
                  f"{(sh.get('0.0') or float('nan')):>7.2f} "
                  f"{(sh.get('0.1') or float('nan')):>7.2f} "
                  f"{(sh.get('0.9') or float('nan')):>7.2f}", flush=True)
        results[key] = rows
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")
    print("READ: real signal => d_maha >> d_mm, stable vs splithalf, and SHUFFLED ~ 0.")
    print("      artifact     => d_maha shrinks under splithalf, or SHUFFLED also large.")


if __name__ == "__main__":
    main()
