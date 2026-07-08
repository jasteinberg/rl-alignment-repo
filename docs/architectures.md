# RL & Alignment Architectures — Reference

The networks and tabular models defined in `utils/`, with the recipe that
trains each one and the notebook (and derivation doc) where it is used.
Grouped by family: tabular control, policy-gradient / PPO, reward modelling,
and probing / calibration diagnostics.

*Drafted with the assistance of Claude (Anthropic).*

_Last updated: 2026-06-18_

---

## Environments (the substrate)

`utils/toy_mdp.py` defines the toy environments the control methods run on:

- **`BanditEnv`** — `k`-armed bandit, Gaussian arm rewards.
- **`HeavyTailedBanditEnv`** — Student-`t` arm rewards (`df` controls tail
  weight); used to stress-test estimators under heavy tails.
- **`GridWorldEnv`** — cliff-walking grid: a start, a goal, and a row of cliff
  cells that reset the agent with a large negative reward. The setting that
  separates on-policy (SARSA) from off-policy (Q-learning) behaviour.

These are not learned; they generate the experience the models below train on.

---

## Tabular control

### TabularPolicy
- **Class:** `utils/toy_mdp.py::TabularPolicy`
- **Definition:** a softmax policy with one logit row per state —
  `logits: nn.Parameter(n_states, n_actions)`; `forward(state)` returns a
  `torch.distributions.Categorical`. No function approximation, so every state
  is independent and the only thing learned is the per-state action preference.
- **Training:** `reinforce(...)` — Monte-Carlo policy gradient, ascending
  `∇_θ log π(a|s) · Â_t`. Derivation in `docs/policy_gradient.md`
  (§3 score function → §5 REINFORCE → §7 natural gradient).
- **Notebook:** `policy_gradient/`.

### TabularValue
- **Class:** `utils/toy_mdp.py::TabularValue`
- **Definition:** `V: nn.Parameter(n_states)` — one scalar value per state.
- **Training:** regressed toward observed returns,
  `V(s) ← V(s) + α_V (G_t − V(s))`, and used as the REINFORCE baseline so the
  advantage `Â_t = G_t − V(s_t)` centres the score (variance reduction,
  `docs/policy_gradient.md` §6).

### SARSA / Q-learning tables
- **Functions:** `utils/toy_mdp.py::sarsa`, `::q_learning`
- **Definition:** not `nn.Module`s — an action-value table `Q(s,a)` updated by
  temporal-difference control. **SARSA** is on-policy (bootstraps on the action
  actually taken, `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') − Q(s,a)]`); **Q-learning**
  is off-policy (bootstraps on the greedy action, `max_{a'} Q(s',a')`). On the
  cliff this is exactly why SARSA learns the safe path and Q-learning the
  risky optimal one.
- **Notebook:** `tabular_control/` (matched-RNG comparison, common-random-numbers
  coupling, GLIE annealing).

---

## Policy gradient / PPO

### NeuralPolicy
- **Class:** `utils/ppo.py::NeuralPolicy`
- **Definition:** MLP actor — `Linear(n_states, 64) → Tanh → Linear(64, n_actions)`;
  `forward(s)` returns a `Categorical` over actions.
- **Training:** `ppo(...)` with `ppo_clip_loss(...)` — the clipped surrogate
  `min(r_t Â_t, clip(r_t, 1±ε) Â_t)`, advantages from the critic. Derivation in
  `docs/ppo.md` (importance ratio → clip → its relation to the natural-gradient
  trust region).
- **Notebook:** `ppo/`.

### NeuralValue
- **Class:** `utils/ppo.py::NeuralValue`
- **Definition:** MLP critic — `Linear(n_states, 64) → Tanh → Linear(64, 1)`;
  estimates `V(s)` for the advantage used by `NeuralPolicy`.
- **Training:** regression to returns / TD targets, jointly with the policy
  inside `ppo(...)`.

---

## Reward modelling

### BradleyTerry
- **Class:** `utils/reward_model.py::BradleyTerry`
- **Definition:** a reward head `Linear(d_input, 1)`; `forward(x)` returns the
  scalar reward `r_φ(x)`. Preferences are modelled as
  `P[y_w ≻ y_l] = σ(r_w − r_l)`.
- **Training:** `train_rm(...)` minimises the Bradley-Terry loss
  `−E[log σ(r_w − r_l)]` (binary cross-entropy on reward margins; method
  `preference_loss`), Adam. `make_synthetic_preferences(...)` builds the
  synthetic `(y_w, y_l)` pairs and `PreferenceDataset` wraps them.
- **Notebooks:** `reward_modeling/`, `rlhf_pipeline/`.

---

## Probing / calibration diagnostics

These are not trained networks — they are closed-form readouts on top of frozen
activations, included here because the notebooks treat them as the "model".

### Linear truth-probe directions
- **Functions:** `utils/probing.py::mass_mean_direction` (+ `direction_snr`,
  `probe_accuracy`).
- **Definition:** a probe direction `θ` separating true/false statements, in
  three closed-form estimators: **plain** mass-mean `θ = μ₁ − μ₀`; **whitened**
  (Fisher/LDA) `θ = Σ⁻¹(μ₁ − μ₀)` via `np.linalg.solve`; **shrink**
  (Ledoit-Wolf) using a shrunk precision for the `n ≪ d` regime. Read out by
  the SNR `d'² = (m₁−m₀)² / [½(s₁²+s₀²)]`.
- **Notebook:** `truth_directions/`.

### Temperature scaling
- **Function:** `utils/calibration.py::temperature_scale`
- **Definition:** a one-parameter post-hoc calibrator — fit a single scalar `T`
  by minimising NLL on held-out logits, then output `σ(z/T)`. Does not change
  accuracy; only the confidence. Evaluated with `ece` (expected calibration
  error).
- **Notebook:** `honesty_empirical/`.
