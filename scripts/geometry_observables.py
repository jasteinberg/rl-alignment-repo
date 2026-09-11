"""
scripts/geometry_observables.py

Order parameters for probe geometry that the interpretability literature does not
report. All are model-free: computed from cached last-token activations.

  PR      participation ratio (sum lam)^2 / sum lam^2 -- the effective number of
          directions carrying within-class variance. PR = 1 means rank-one.
  cos_v1  |cos(theta_hat, v_1)|, alignment of the probe with the dominant
          ("rogue" / massive-activation) direction.
  d_mm    mass-mean separation |delta.theta| / sqrt(pooled var along theta)
  d_maha  full Mahalanobis separation sqrt(delta^T Sigma^{-1} delta), i.e. the
          separation available to an optimal linear readout. d_maha / d_mm
          measures how much signal the naive estimator leaves on the table.
  var_th  Var(x . theta) -- the spontaneous fluctuation along the probe. Paired
          with the steering susceptibility chi it tests a fluctuation-response
          relation: in an equilibrium system response ~ fluctuation, and a
          transformer has no reason to obey that.
  overlap |cos(theta_L, theta_L')| across layers -- is the direction propagated
          through depth, or recomputed?

Drafted with the assistance of Claude (Anthropic).
"""
import os, json, glob, argparse
import numpy as np
from truthlib.estimators import observables

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "artifacts", "act_cache")
OUT = os.path.join(REPO, "artifacts", "geometry_observables.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-shrink", dest="shrink", action="store_false")
    args = ap.parse_args()

    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        model, ds = key.split("__")
        z = np.load(p); y = z["y"]
        layers = sorted([int(k[1:]) for k in z.files if k.startswith("L")])
        print(f"\n=== {model} / {ds} ===", flush=True)
        print(f"{'L':>4} {'PR':>8} {'cos_v1':>8} {'d_mm':>7} {'d_maha':>8} "
              f"{'hidden':>7} {'var_th':>11}", flush=True)
        per_layer, thetas = {}, {}
        for L in layers:
            o = observables(z[f"L{L}"].astype(np.float64), y, args.shrink)
            thetas[L] = np.array(o.pop("theta"))
            per_layer[str(L)] = o
            print(f"{L:>4} {o['PR']:>8.2f} {o['cos_theta_v1']:>8.3f} "
                  f"{o['d_mass_mean']:>7.3f} {o['d_mahalanobis']:>8.2f} "
                  f"{o['hidden_signal_ratio']:>7.1f} {o['var_along_theta']:>11.2f}",
                  flush=True)

        # layer-to-layer overlap of the probe direction
        ov = {}
        for L in layers:
            ov[str(L)] = {str(L2): float(abs(thetas[L] @ thetas[L2])) for L2 in layers}
        print("\n  layer overlap |cos(theta_L, theta_L')|", flush=True)
        print("      " + "".join(f"{L:>7}" for L in layers), flush=True)
        for L in layers:
            print(f"  L{L:<3}" + "".join(f"{ov[str(L)][str(L2)]:>7.2f}" for L2 in layers),
                  flush=True)

        results[key] = {"model": model, "dataset": ds,
                        "layers": per_layer, "layer_overlap": ov}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
