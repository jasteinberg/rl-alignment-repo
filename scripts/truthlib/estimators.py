"""
Estimators for the truth-directions analysis.

NumPy and scikit-learn only, no torch, so every figure and table can be rebuilt
from the committed JSON on any machine. Each function reproduces, operation for
operation, the arithmetic of the copies it replaces; where those copies
disagreed, the choice is an explicit argument rather than a silent default.

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


# ---- splitting and centring ---------------------------------------------
def split_indices(y, frac=0.5, seed=0):
    """Class-stratified train/test split (seed 0 is the post's split)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        tr.append(idx[:k]); te.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)


def within_class_center(X, y):
    """Centre each class on its own mean (returns data, not a covariance)."""
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] = X[y == lab] - X[y == lab].mean(0)
    return Xc


def within_class_cov(X, y, ddof=0):
    """Within-class covariance C-hat.

    ddof=0 divides by N, the post's definition and the Ledoit-Wolf convention.
    ddof=1 is np.cov, used by geometry_observables, check_rogue_dimension and
    the cover_* scripts; kept so those artifacts reproduce exactly. Ratios
    (lambda1/tr, lambda1/lambda2, PR) and eigenvectors are identical under both.
    """
    Xc = within_class_center(X, y)
    if ddof == 1:
        return np.cov(Xc, rowvar=False)
    return (Xc.T @ Xc) / len(y)


# ---- separation -----------------------------------------------------------
def d_prime(z, y):
    """Held-out d' of a projection z: |m1 - m0| / sqrt(pooled variance).

    Returns 0 when the pooled variance vanishes (the layer-0 embedding on
    templated sets), which is what makes those rows read AUROC 0.500.
    """
    a, b = z[y == 1], z[y == 0]
    den = 0.5 * (a.var(ddof=1) + b.var(ddof=1))
    return float(abs(a.mean() - b.mean()) / np.sqrt(den)) if den > 0 else 0.0


def auroc(z, y):
    return float(roc_auc_score(y, z))


# ---- directions -----------------------------------------------------------
def class_gap(X, y):
    """||mu1 - mu0||, the steering unit c: a property of dataset and layer,
    never of the direction (std(X @ theta) was the extend_null scale bug)."""
    return float(np.linalg.norm(X[y == 1].mean(0) - X[y == 0].mean(0)))


def mass_mean(X, y):
    """theta-hat = delta-hat / ||delta-hat||."""
    t = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def fisher(X, y, rho=None):
    """Whitened direction Sigma-hat^{-1} delta-hat, unit norm.

    rho=None: sklearn Ledoit-Wolf precision, the post's theta-hat_F.
    rho=float: explicit shrinkage (1-rho) C + rho (tr C / d) I, C divided by N,
    solved rather than inverted. The two paths agree to floating point, not
    bitwise, at rho = rho_LW.
    """
    Xc = within_class_center(X, y)
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    if rho is None:
        t = LedoitWolf(assume_centered=True).fit(Xc).precision_ @ dlt
    else:
        C = (Xc.T @ Xc) / len(y)
        mu = np.trace(C) / C.shape[0]
        S = (1.0 - rho) * C + rho * mu * np.eye(C.shape[0])
        t = np.linalg.solve(S, dlt)
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


# ---- massive coordinates ------------------------------------------------------
def massive_mask(X, mag=100.0, rel=100.0, frac=0.5):
    """Flag statements that LACK the near-universal massive activation.

    Coordinate j is massive when |med_i x_ij| exceeds `mag` and `rel` times the
    median absolute activation (Sun et al. use 1000x; relaxed to 100x because the
    dominant coordinate runs only 564-587x the median at pythia-2.8b L24-L28,
    and every value from 100x to 560x flags the same eleven). Statement i is a
    dropper when |x_ij - med_j| > frac |med_j| for some massive j. Never touches
    v1, the class means, or the labels.

    Returns (dropped, med_all, massive).
    """
    A = np.abs(X)
    med_all = float(np.median(A))
    med_j = np.median(X, axis=0)
    massive = (np.abs(med_j) > mag) & (np.abs(med_j) > rel * med_all)
    if not massive.any():
        return np.zeros(len(X), bool), med_all, massive
    dev = np.abs(X[:, massive] - med_j[massive]) > frac * np.abs(med_j[massive])
    return dev.any(1), med_all, massive


# ---- spectra and steering summaries ---------------------------------------------
def participation_ratio(lam):
    """(sum lam)^2 / sum lam^2: the number of directions the spectrum occupies."""
    tr = lam.sum()
    return float(tr ** 2 / (lam ** 2).sum())


def chi_origin(alphas, values):
    """Steering susceptibility: least-squares slope of A against h through 0."""
    a = np.array(alphas, dtype=float)
    v = np.array(values, dtype=float)
    return float((a * v).sum() / (a * a).sum())
