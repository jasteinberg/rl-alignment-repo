# rl-alignment-repo

RL theory, RLHF pipelines, and alignment/honesty experiments built from
first principles at laptop scale.

Author: Julia Steinberg

*Drafted with the assistance of Claude (Anthropic).*

## Repo layout

```
rl-alignment-repo/
├── utils/        # Toy MDP, PPO, reward model (Bradley-Terry),
│                 #   calibration, linear probing
├── scripts/      # Geometry-of-truth probing experiments (SNR analysis
│                 #   of truth directions across models/datasets)
├── notebooks/    # Executed notebooks, one per topic:
│   ├── tabular_control/      # SARSA vs Q-learning
│   ├── policy_gradient/      # REINFORCE on gridworld
│   ├── ppo/                  # PPO + trust region
│   ├── reward_modeling/      # Bradley-Terry reward models
│   ├── rlhf_pipeline/        # Full pipeline, KL penalties, hacking
│   ├── scalable_oversight/   # Weak-to-strong generalization
│   ├── honesty_empirical/    # Calibration, sycophancy
│   └── truth_directions/     # Truth-direction probing
├── artifacts/    # Truth-direction SNR grid figures
└── docs/
```
