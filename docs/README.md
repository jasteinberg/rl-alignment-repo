# docs/ — guide

What each reference doc covers, and how docs and notebooks line up.

## Reference
- **architectures.md** — every model defined in `utils/` and how it is trained
  (tabular policy/value, SARSA/Q tables, PPO actor/critic MLPs, the Bradley-Terry
  reward model, truth probes, temperature scaling), grouped by family.
- **reproducing.md** — how to run from a clone (most notebooks are self-contained
  synthetic; `truth_directions` needs the Geometry-of-Truth data + model downloads).
- **external_sources.md** — reading lists and dataset/source attributions.

## Derivation companions (physics-textbook register, full derivations)
- **policy_gradient.md** — MDP & objective → score function → policy-gradient
  theorem → REINFORCE → baselines/advantage → natural gradient. Backs
  `notebooks/policy_gradient`.
- **ppo.md** — importance sampling → ratio explosion → clipped surrogate → its
  relation to the TRPO / natural-gradient trust region. Backs `notebooks/ppo`.

## Notebook → doc map

| notebook | docs |
|---|---|
| `policy_gradient` | policy_gradient, architectures |
| `ppo` | ppo, architectures |
| `tabular_control` | architectures |
| `reward_modeling` | architectures |
| `rlhf_pipeline` | architectures (KL / free-energy derivation is in-notebook) |
| `scalable_oversight` | architectures |
| `truth_directions` | architectures, external_sources |
| `honesty_empirical` | architectures |
