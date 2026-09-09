"""
In-sample versus held-out d' on a synthetic control with a planted separation.

The post quotes a synthetic control in *Setup*: "with a planted separation of
d' = 1, the in-sample estimate returns 1.31 and the held-out estimate 0.58".
No script produced those numbers and they are not reproducible at the post's
own d/N. This script does the control at the post's regime and at a ladder of
d/N so the quoted sentence can carry numbers that exist.

Model: x ~ N(+-delta/2, I_d), delta = u (unit), so the population d' along u
is exactly 1. Mass-mean direction fit on a class-balanced training half, scored
in-sample on that half and held-out on the other. Prediction from the
shuffled-label derivation in the post: in-sample d'^2 ~ 1 + 4d/N_train (the
noise part of theta-hat is aligned with the sample fluctuations that defined
it), held-out d' ~ cos(theta-hat, u) ~ 1/sqrt(1 + 4d/N_train).

Drafted with the assistance of Claude (Anthropic).
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "check_insample_attenuation.json"
N_REP = 20


def dprime(X, y, th):
    p = X @ th
    p1, p0 = p[y == 1], p[y == 0]
    return float(abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1))))


def control(d, N, rng):
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    y = np.r_[np.zeros(N // 2), np.ones(N // 2)].astype(int)
    X = rng.standard_normal((N, d)) + np.outer(y - 0.5, u)
    perm = rng.permutation(N)
    tr, te = perm[: N // 2], perm[N // 2:]
    th = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    th /= np.linalg.norm(th)
    return dprime(X[tr], y[tr], th), dprime(X[te], y[te], th), float(abs(th @ u))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for d, N in [(2560, 1198), (2048, 1198), (1024, 1198), (512, 1198), (2560, 354), (128, 1198)]:
        r = np.array([control(d, N, rng) for _ in range(N_REP)])
        ntr = N // 2
        pred_in = float(np.sqrt(1 + 4 * d / ntr))
        pred_out = float(1 / np.sqrt(1 + 4 * d / ntr))
        rows.append({"d": d, "N": N, "N_train": ntr, "d_over_Ntrain": d / ntr,
                     "insample_mean": float(r[:, 0].mean()), "insample_sd": float(r[:, 0].std(ddof=1)),
                     "heldout_mean": float(r[:, 1].mean()), "heldout_sd": float(r[:, 1].std(ddof=1)),
                     "cos_mean": float(r[:, 2].mean()),
                     "pred_insample": pred_in, "pred_heldout": pred_out})
        print(f"d={d:5d} N={N:5d} d/Ntr={d/ntr:5.2f}  in-sample {r[:,0].mean():.2f}±{r[:,0].std():.2f} (pred {pred_in:.2f})"
              f"  held-out {r[:,1].mean():.2f}±{r[:,1].std():.2f} (pred {pred_out:.2f})  cos={r[:,2].mean():.2f}")
    OUT.write_text(json.dumps({"planted_dprime": 1.0, "n_rep": N_REP, "rows": rows}, indent=2))
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    main()
