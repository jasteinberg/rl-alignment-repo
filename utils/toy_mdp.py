"""
utils/toy_mdp.py
================
Simple toy MDPs for testing and illustrating RL concepts.

Includes:
  - BanditEnv             : k-armed Gaussian bandit
  - HeavyTailedBanditEnv  : k-armed bandit with Student-t reward noise
  - GridWorldEnv          : finite tabular MDP on a 2D grid
  - TabularPolicy         : softmax policy over (state, action) table
  - reinforce             : REINFORCE with baseline (policy gradient)
  - mc_control            : every-visit Monte Carlo control
  - sarsa                 : on-policy TD(0) control
  - q_learning            : off-policy TD(0) control
  - make_episode_rng_factory : shared per-episode RNG for CRN comparisons
  - ExplorationTape       : timestep-indexed CRN tape (coin / explore-action)
  - make_tape_factory     : per-episode ExplorationTape factory

Usage:
    from utils.toy_mdp import GridWorldEnv, TabularPolicy, reinforce
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Callable


# ── k-armed Bandit ────────────────────────────────────────────────────────────

class BanditEnv:
    """
    Stateless k-armed Gaussian bandit.
    Useful for isolating the reward signal from transition dynamics.

    True means: q* ~ N(0, 1). Each pull: r ~ N(q*_a, 1).
    """
    def __init__(self, k: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.q_star = rng.standard_normal(k)
        self.k = k
        self.optimal = int(np.argmax(self.q_star))

    def step(self, action: int) -> float:
        return float(np.random.normal(self.q_star[action], 1.0))


# ── Heavy-tailed Bandit ───────────────────────────────────────────────────────

class HeavyTailedBanditEnv:
    """
    Stateless k-armed bandit with Student-t reward noise.
    Useful for stress-testing estimators when reward variance is large or undefined.

    True means: q* ~ N(0, 1). Each pull: r = q*_a + sigma * T_df,
    where T_df is a standard Student-t variate.
        df = 1  -> Cauchy (undefined mean)
        df = 2  -> finite mean, infinite variance
        df > 2  -> finite variance = df / (df - 2)
    """
    def __init__(self, k: int = 10, df: float = 3.0, sigma: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.q_star = rng.standard_normal(k)
        self.k = k
        self.df = df
        self.sigma = sigma
        self.optimal = int(np.argmax(self.q_star))

    def step(self, action: int) -> float:
        return float(self.q_star[action] + self.sigma * np.random.standard_t(self.df))


# ── GridWorld ─────────────────────────────────────────────────────────────────

class GridWorldEnv:
    """
    Finite tabular MDP on an (H x W) grid.

    States:  (row, col) flattened to integer s = row*W + col
    Actions: 0=up, 1=down, 2=left, 3=right
    Reward:  +1 on reaching goal, -0.01 per step, 0 otherwise
    Terminal: goal state or max_steps reached

    Walls: list of (row, col) cells the agent cannot enter.
    """
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up down left right

    def __init__(
        self,
        H: int = 4,
        W: int = 4,
        goal: Optional[Tuple[int, int]] = None,
        start: Tuple[int, int] = (0, 0),
        walls: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 100,
        gamma: float = 0.99,
    ):
        self.H, self.W = H, W
        self.n_states  = H * W
        self.n_actions = 4
        self.goal      = goal if goal is not None else (H - 1, W - 1)
        self.start     = start
        self.walls     = set(walls or [])
        self.max_steps = max_steps
        self.gamma     = gamma
        self.reset()

    def _flat(self, r, c): return r * self.W + c
    def _rc(self, s):      return divmod(s, self.W)

    def reset(self) -> int:
        self.pos  = self.start
        self.step_count = 0
        return self._flat(*self.pos)

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = self.MOVES[action]
        r, c   = self.pos
        nr, nc = r + dr, c + dc

        # Clip to grid and respect walls
        if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
            self.pos = (nr, nc)

        self.step_count += 1
        done   = (self.pos == self.goal) or (self.step_count >= self.max_steps)
        reward = 1.0 if self.pos == self.goal else -0.01
        return self._flat(*self.pos), reward, done


# ── Tabular Softmax Policy ────────────────────────────────────────────────────

class TabularPolicy(nn.Module):
    """
    Softmax policy: pi(a|s) = softmax(theta[s, :]).
    theta is a learnable (n_states, n_actions) preference table.
    """
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_states, n_actions))

    def forward(self, state: int) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits[state])

    def action_and_logprob(self, state: int) -> Tuple[int, torch.Tensor]:
        dist = self(state)
        a    = dist.sample()
        return a.item(), dist.log_prob(a)


# ── Value function baseline ───────────────────────────────────────────────────

class TabularValue(nn.Module):
    """Learnable state value table V(s) used as REINFORCE baseline."""
    def __init__(self, n_states: int):
        super().__init__()
        self.V = nn.Parameter(torch.zeros(n_states))

    def forward(self, state: int) -> torch.Tensor:
        return self.V[state]


# ── REINFORCE with baseline ───────────────────────────────────────────────────

def reinforce(
    env: GridWorldEnv,
    policy: TabularPolicy,
    value: TabularValue,
    n_episodes: int = 2000,
    lr_policy: float = 1e-2,
    lr_value:  float = 1e-2,
    gamma:     float = 0.99,
) -> dict:
    """
    REINFORCE with state-value baseline.

    Policy gradient update (per timestep t):
        grad += grad_log_pi(a_t|s_t) * (G_t - V(s_t))

    Baseline update (MSE on returns):
        V(s_t) <- V(s_t) + lr * (G_t - V(s_t))

    Returns history dict with episode returns and losses.
    """
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    opt_v  = torch.optim.Adam(value.parameters(),  lr=lr_value)
    history = {"returns": [], "pg_loss": [], "v_loss": []}

    for ep in range(n_episodes):
        # ── Rollout ──────────────────────────────────────────────────────────
        states, log_probs, rewards = [], [], []
        s = env.reset()
        done = False
        while not done:
            a, lp = policy.action_and_logprob(s)
            s2, r, done = env.step(a)
            states.append(s); log_probs.append(lp); rewards.append(r)
            s = s2

        # ── Discounted returns G_t ────────────────────────────────────────
        T  = len(rewards)
        G  = torch.zeros(T)
        g  = 0.0
        for t in reversed(range(T)):
            g    = rewards[t] + gamma * g
            G[t] = g

        # ── Baseline ──────────────────────────────────────────────────────
        V_s = torch.stack([value(s) for s in states])
        adv = (G - V_s.detach())

        # ── Policy gradient loss ───────────────────────────────────────────
        log_probs_t = torch.stack(log_probs)
        pg_loss     = -(log_probs_t * adv).mean()

        opt_pi.zero_grad(); pg_loss.backward(); opt_pi.step()

        # ── Value loss (MSE) ───────────────────────────────────────────────
        v_loss = F.mse_loss(V_s, G)
        opt_v.zero_grad(); v_loss.backward(); opt_v.step()

        history["returns"].append(float(G[0]))
        history["pg_loss"].append(float(pg_loss))
        history["v_loss"].append(float(v_loss))

    return history


# ── Value-based tabular methods ───────────────────────────────────────────────
# The methods below use numpy rather than torch: tabular bootstrap updates have
# closed-form table writes, no autograd needed.

def epsilon_greedy(
    Q_s: np.ndarray,
    epsilon: float,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """
    Sample an action: uniform random w.p. epsilon, else argmax of Q[s].

    rng:
        None → draws use the module-global numpy RNG (legacy behaviour).
        np.random.RandomState → draws come from this stream. Pass the SAME
        RandomState to two algorithms and they share (coin, random-action)
        decisions step-by-step — common random numbers (CRN).
    """
    n_actions = len(Q_s)
    if rng is None:
        if np.random.random() < epsilon:
            return int(np.random.randint(n_actions))
        return int(np.argmax(Q_s))
    if rng.random() < epsilon:
        return int(rng.randint(n_actions))
    return int(np.argmax(Q_s))


def resolve_epsilon(epsilon, ep_idx: int) -> float:
    """
    Resolve an epsilon specification to a scalar for a given episode.

    epsilon may be:
      - a float          → constant exploration rate (returned as-is).
      - a callable(int)  → an annealing schedule; called with ep_idx.

    Lets sarsa()/q_learning()/mc_control() accept either a fixed epsilon or
    a GLIE-style decay schedule without changing their call signatures.
    """
    eps = epsilon(ep_idx) if callable(epsilon) else epsilon
    return float(eps)


def geometric_epsilon(eps0: float, decay: float, eps_min: float = 0.0):
    """
    Build a geometric epsilon-decay schedule: episode k -> max(eps_min,
    eps0 * decay**k).

    Note: with eps_min > 0 this is not strictly GLIE (which requires
    epsilon -> 0). It is the common practical form; set eps_min = 0 for a
    schedule that does satisfy the GLIE limit condition.
    """
    def schedule(ep_idx: int) -> float:
        return max(eps_min, eps0 * (decay ** ep_idx))
    return schedule


def harmonic_epsilon(eps0: float, c: float):
    """
    Build a harmonic epsilon-decay schedule: episode k -> eps0 / (1 + c*k).

    The distinction from geometric_epsilon matters for asymptotic
    convergence. GLIE requires epsilon -> 0 while every state-action pair
    is still visited infinitely often. A policy that avoids a region is
    driven into it only by exploration, at a per-episode rate ~ epsilon_k.
    Whether that region is visited infinitely often is governed by whether
    sum_k epsilon_k diverges (Borel-Cantelli):

      - geometric, epsilon_k ~ decay**k : sum converges -> avoided regions
        are visited only finitely often -> their Q-values never converge ->
        GLIE visitation condition VIOLATED.
      - harmonic,  epsilon_k ~ 1/k      : sum diverges -> avoided regions
        are still visited infinitely often -> GLIE condition SATISFIED.

    Harmonic decay is therefore the schedule that supports the Sutton &
    Barto asymptotic-convergence guarantee; geometric decay does not, even
    though it looks like a reasonable anneal. The price is slow decay.
    """
    def schedule(ep_idx: int) -> float:
        return eps0 / (1.0 + c * ep_idx)
    return schedule


def make_episode_rng_factory(master_seed: int) -> Callable[[int], np.random.RandomState]:
    """
    Build a callable factory(ep_idx) -> np.random.RandomState seeded
    deterministically from (master_seed, ep_idx). Pass the same factory
    to sarsa() and q_learning() to share exploration streams per episode:
    both algorithms then consume identical (coin, random-action) draws
    in step-order within each episode.

    Determinism: hash((master_seed, ep_idx)) masked to 31 bits. Order-
    independent — the factory can be called in any order, any number of
    times, by either algorithm, and returns equivalent streams for the
    same ep_idx.
    """
    def factory(ep_idx: int) -> np.random.RandomState:
        seed_val = hash((master_seed, int(ep_idx))) & 0x7FFFFFFF
        return np.random.RandomState(seed_val)
    return factory


class ExplorationTape:
    """
    Timestep-indexed common-random-numbers tape for epsilon-greedy control.

    An epsilon-greedy step draws two independent randoms: a coin c_tau in
    [0, 1) deciding explore-vs-exploit, and (used only when exploring) a
    uniform action u_tau. This tape stores the pair (c_tau, u_tau) keyed by
    the timestep tau of the action that is *executed* at that step — NOT by
    the order in which an algorithm happens to call it.

    Why timestep-indexed: SARSA's epsilon-greedy call at the end of step t
    selects the action executed at step t+1 (plus a priming call for step
    0); Q-learning's call at step t selects the action for step t. Indexing
    by executed-step rather than call-order means both algorithms read the
    SAME (c_tau, u_tau) for the action at step tau, despite the call-timing
    asymmetry between them.

    Lazy fill makes it order-independent: each slot tau derives its own RNG
    from (seed, tau), so tape.get(tau) returns the same pair no matter when
    or in what order it is first requested, by either algorithm. This is
    what makes the tape robust — accessing slot 5 before slot 2 does not
    perturb slot 2. (An earlier design drew from a single shared RNG in
    call-order; that made slot values depend on access order and is a
    debugging hazard. Per-slot seeding removes the hazard entirely.)

    Coupling control via share_explore_action:
        True  → both algorithms use the tape's u_tau on explore steps
                (full common random numbers: identical exploratory action).
        False → the coin c_tau is still shared (explore/exploit schedule is
                locked), but each algorithm draws its own explore action
                from its own RNG. Use when the comparison should marginalise
                over exploration *content* while fixing the *schedule*.

    Note there is no "share action but not coin" mode: it is incoherent
    (if the coins differ, only one algorithm is exploring that step, so
    there is no shared action to speak of). Fully decoupled exploration is
    obtained simply by NOT using a tape — i.e. the legacy global-RNG path.
    """

    def __init__(self, seed: int, n_actions: int):
        self._seed = int(seed)
        self._n_actions = n_actions
        self._tape: dict = {}          # tau -> (coin, explore_action)

    def get(self, tau: int) -> Tuple[float, int]:
        """Return (coin, explore_action) for the action executed at step tau.

        Order-independent: slot tau's value derives from (seed, tau) alone.
        """
        tau = int(tau)
        if tau not in self._tape:
            slot_rng = np.random.RandomState(
                hash((self._seed, tau)) & 0x7FFFFFFF
            )
            self._tape[tau] = (slot_rng.random(),
                               int(slot_rng.randint(self._n_actions)))
        return self._tape[tau]

    def reset(self) -> None:
        """Clear the cached tape (slot values on re-fill are unchanged)."""
        self._tape.clear()


def make_tape_factory(
    master_seed: int,
    n_actions: int,
) -> Callable[[int], ExplorationTape]:
    """
    Build a callable factory(ep_idx) -> ExplorationTape, one fresh tape per
    episode, seeded deterministically from (master_seed, ep_idx).

    Pass the same factory to sarsa() and q_learning(): within each episode
    both algorithms read identical (coin, explore-action) pairs keyed by
    executed-timestep. See ExplorationTape for the coupling semantics.
    """
    def factory(ep_idx: int) -> ExplorationTape:
        seed_val = hash((master_seed, int(ep_idx))) & 0x7FFFFFFF
        return ExplorationTape(seed_val, n_actions)
    return factory


def epsilon_greedy_taped(
    Q_s: np.ndarray,
    epsilon: float,
    coin: float,
    explore_action: int,
    own_rng: Optional[np.random.RandomState] = None,
    share_explore_action: bool = True,
) -> int:
    """
    epsilon-greedy selection driven by a pre-drawn coin from an ExplorationTape.

    coin:            the shared Bernoulli draw c_tau (explore iff coin < epsilon).
    explore_action:  the tape's shared action u_tau (used iff sharing actions).
    own_rng:         per-algorithm RNG, used to draw an independent explore
                     action when share_explore_action is False. Required in
                     that mode; ignored otherwise.
    share_explore_action:
                     True  → return the tape's explore_action on explore steps.
                     False → draw an independent explore action from own_rng.

    The exploit branch always returns argmax(Q_s): the learned policies differ
    between algorithms, and that difference is the object of study.
    """
    if coin < epsilon:
        if share_explore_action:
            return int(explore_action)
        if own_rng is None:
            raise ValueError(
                "share_explore_action=False requires own_rng to draw an "
                "independent explore action."
            )
        return int(own_rng.randint(len(Q_s)))
    return int(np.argmax(Q_s))


# ── Monte Carlo Control ───────────────────────────────────────────────────────

def mc_control(
    env: GridWorldEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    alpha: float = 0.1,
) -> dict:
    """
    Every-visit Monte Carlo control with epsilon-greedy exploration.

    Update (per visit to (s_t, a_t) in the episode):
        Q(s_t, a_t) <- Q(s_t, a_t) + alpha * (G_t - Q(s_t, a_t))

    Constant step-size alpha (rather than 1/N) keeps the update form aligned
    with TD methods for direct comparison.
    """
    Q = np.zeros((env.n_states, env.n_actions))
    history = {"returns": []}

    for ep in range(n_episodes):
        eps = resolve_epsilon(epsilon, ep)
        # ── Rollout ──────────────────────────────────────────────────────────
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q[s], eps)
            s2, r, done = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = s2

        # ── Returns + every-visit update ─────────────────────────────────────
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            s_t, a_t = states[t], actions[t]
            Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])

        history["returns"].append(sum(rewards))

    return {"Q": Q, **history}


# ── SARSA (on-policy TD(0)) ───────────────────────────────────────────────────

def sarsa(
    env: GridWorldEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    rng_factory: Optional[Callable[[int], np.random.RandomState]] = None,
    tape_factory: Optional[Callable[[int], "ExplorationTape"]] = None,
    share_explore_action: bool = True,
    record_trajectories: bool = False,
) -> dict:
    """
    On-policy TD(0) control.

    Update:
        Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
    where a' is sampled from the same epsilon-greedy policy that generated a.

    Three mutually exclusive exploration sources (pick at most one of the
    two factory arguments):

    rng_factory: callable(ep_idx) -> np.random.RandomState. Each episode's
        epsilon-greedy draws come from a fresh per-episode RNG, consumed in
        call-order. Pair with q_learning(rng_factory=same_factory) for a
        common-random-numbers comparison. Correct for SARSA-vs-Q-learning
        because their k-th draw happens to be step k for both; not robust
        to algorithms with a different call pattern.

    tape_factory: callable(ep_idx) -> ExplorationTape. Each episode's draws
        are keyed by the *executed timestep* rather than call-order, so the
        matching survives any call-timing asymmetry. Use make_tape_factory()
        to build one. This is the recommended path; see ExplorationTape.

    share_explore_action: only relevant when tape_factory is set. True →
        both algorithms take the tape's shared action on explore steps.
        False → the coin (explore/exploit schedule) is still shared, but
        each algorithm draws its own explore action. Ignored otherwise.

    If neither factory is given, draws use the module-global numpy RNG
    (legacy behaviour — fully decoupled when two algorithms are run
    separately).

    record_trajectories: if True, the returned dict has an extra key
        'trajectories' whose value is a list of n_episodes lists of
        (state, action) pairs — the actions actually executed each step.
    """
    if rng_factory is not None and tape_factory is not None:
        raise ValueError("Pass at most one of rng_factory, tape_factory.")

    Q = np.zeros((env.n_states, env.n_actions))
    history: dict = {"returns": []}
    if record_trajectories:
        history["trajectories"] = []

    for ep in range(n_episodes):
        eps = resolve_epsilon(epsilon, ep)
        rng = rng_factory(ep) if rng_factory is not None else None
        tape = tape_factory(ep) if tape_factory is not None else None
        # own_rng: independent explore-action draws when not sharing actions.
        own_rng = (np.random.RandomState(hash(("sarsa", ep)) & 0x7FFFFFFF)
                   if (tape is not None and not share_explore_action) else None)

        def pick(Q_s: np.ndarray, tau: int) -> int:
            """Select an action for executed-step tau via the active source."""
            if tape is not None:
                coin, u = tape.get(tau)
                return epsilon_greedy_taped(Q_s, eps, coin, u,
                                            own_rng=own_rng,
                                            share_explore_action=share_explore_action)
            return epsilon_greedy(Q_s, eps, rng=rng)

        s = env.reset()
        a = pick(Q[s], 0)                       # priming call → executed step 0
        traj = [(s, a)] if record_trajectories else None
        done = False
        ep_return = 0.0
        t = 0
        while not done:
            s2, r, done = env.step(a)
            # in-loop call selects a2, the action executed at step t+1
            a2 = pick(Q[s2], t + 1) if not done else 0
            target = r + (0.0 if done else gamma * Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            ep_return += r
            t += 1
            if record_trajectories and not done:
                traj.append((s, a))
        history["returns"].append(ep_return)
        if record_trajectories:
            history["trajectories"].append(traj)

    return {"Q": Q, **history}


# ── Q-learning (off-policy TD(0)) ─────────────────────────────────────────────

def q_learning(
    env: GridWorldEnv,
    n_episodes: int = 2000,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    rng_factory: Optional[Callable[[int], np.random.RandomState]] = None,
    tape_factory: Optional[Callable[[int], "ExplorationTape"]] = None,
    share_explore_action: bool = True,
    record_trajectories: bool = False,
) -> dict:
    """
    Off-policy TD(0) control.

    Update:
        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
    Behavior policy: epsilon-greedy. Target policy: greedy (the max).

    rng_factory, tape_factory, share_explore_action, record_trajectories:
    see sarsa(). Pair with sarsa(tape_factory=same_factory) for a
    common-random-numbers comparison robust to the call-timing asymmetry
    between the two algorithms.
    """
    if rng_factory is not None and tape_factory is not None:
        raise ValueError("Pass at most one of rng_factory, tape_factory.")

    Q = np.zeros((env.n_states, env.n_actions))
    history: dict = {"returns": []}
    if record_trajectories:
        history["trajectories"] = []

    for ep in range(n_episodes):
        eps = resolve_epsilon(epsilon, ep)
        rng = rng_factory(ep) if rng_factory is not None else None
        tape = tape_factory(ep) if tape_factory is not None else None
        own_rng = (np.random.RandomState(hash(("q_learning", ep)) & 0x7FFFFFFF)
                   if (tape is not None and not share_explore_action) else None)

        def pick(Q_s: np.ndarray, tau: int) -> int:
            """Select an action for executed-step tau via the active source."""
            if tape is not None:
                coin, u = tape.get(tau)
                return epsilon_greedy_taped(Q_s, eps, coin, u,
                                            own_rng=own_rng,
                                            share_explore_action=share_explore_action)
            return epsilon_greedy(Q_s, eps, rng=rng)

        s = env.reset()
        traj = [] if record_trajectories else None
        done = False
        ep_return = 0.0
        t = 0
        while not done:
            a = pick(Q[s], t)                   # step-t call → executed step t
            if record_trajectories:
                traj.append((s, a))
            s2, r, done = env.step(a)
            target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            ep_return += r
            t += 1
        history["returns"].append(ep_return)
        if record_trajectories:
            history["trajectories"].append(traj)

    return {"Q": Q, **history}
