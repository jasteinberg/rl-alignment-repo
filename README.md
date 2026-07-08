# rl-alignment-repo

RL theory, RLHF pipelines, and alignment/honesty experiments built from
first principles at laptop scale.

Author: Julia Steinberg

*Drafted with the assistance of Claude (Anthropic).*

This repo is being populated incrementally. The reusable machinery (`utils/`)
and the truth-direction probing study (`scripts/`) are in place, and the worked
notebooks (`notebooks/`) that exercise each piece are being added one at a time —
starting with tabular control and policy gradient; see [Coming soon](#coming-soon) for the rest.

## What's here

```
rl-alignment-repo/
├── utils/        # From-scratch RL/alignment building blocks:
│                 #   toy MDP + tabular control, PPO (clipped, trust region),
│                 #   Bradley-Terry reward model, training loop,
│                 #   linear probing, calibration (ECE)
├── scripts/      # Truth-direction SNR study: fetch/generate true-false
│                 #   statement data and measure the signal-to-noise ratio
│                 #   of linear "truth directions" across models and datasets
├── notebooks/    # Worked notebooks on the utils/ machinery:
│                 #   tabular control (SARSA vs Q), policy gradient (REINFORCE)
└── docs/         # Reference + full-derivation companions to the notebooks
```

- **`utils/`** — core implementations (`toy_mdp.py`, `ppo.py`, `reward_model.py`,
  `train.py`, `probing.py`, `calibration.py`), each written to stand alone and
  run on a laptop; paths resolve from the repo root so the code runs wherever
  it's checked out.
- **`scripts/`** — geometry-of-truth probing: `make_truth_data.py` builds
  self-contained true/false datasets, `fetch_geometry_of_truth.py` pulls the
  Marks & Tegmark (2023) sets, and `snr_grid.py` / `truth_direction_snr.py`
  compute the truth-direction SNR across models and datasets.
- **`notebooks/`** — worked notebooks that exercise the `utils/` code end to end.
  So far: `tabular_control/` — SARSA vs Q-learning on cliff-walking as a properly
  controlled experiment (named RNG streams, common random numbers, the paired
  difference $\Delta = R_S - R_Q$, behavior vs greedy-evaluation return); and
  `policy_gradient/` — REINFORCE on gridworld, from the score-function identity
  through baselines and the advantage view.
- **`docs/`** — reference and full-derivation companions to the notebooks
  (physics-textbook register), with a notebook↔doc map in `docs/README.md`.

## Coming soon

Further worked notebooks built on the `utils/` machinery:

- **PPO** — clipping and the trust-region view
- **reward modeling** — Bradley-Terry reward models
- **RLHF pipeline** — full pipeline, KL penalties, reward hacking
- **scalable oversight** — weak-to-strong generalization
- **honesty (empirical)** — calibration and sycophancy
- **truth directions** — probing write-up with the SNR figures
