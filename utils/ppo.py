"""
utils/ppo.py
============
PPO (Proximal Policy Optimization) on tabular and neural policies.

Covers:
  - Importance sampling ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
  - Clipped surrogate objective L^CLIP
  - PPO training loop with value network

Usage:
    from utils.ppo import NeuralPolicy, NeuralValue, ppo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


# ── Neural policy (MLP) ───────────────────────────────────────────────────────

class NeuralPolicy(nn.Module):
    """
    Small MLP policy: one-hot state -> action logits.
    Replaces the tabular policy for PPO to make importance sampling meaningful.
    """
    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self._n_states = n_states

    def _one_hot(self, s: int) -> torch.Tensor:
        x = torch.zeros(self._n_states)
        x[s] = 1.0
        return x

    def logits(self, s: int) -> torch.Tensor:
        return self.net(self._one_hot(s))

    def forward(self, s: int) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(s))

    def action_and_logprob(self, s: int):
        dist = self(s)
        a    = dist.sample()
        return a.item(), dist.log_prob(a)

    def log_prob(self, s: int, a: int) -> torch.Tensor:
        return self(s).log_prob(torch.tensor(a))


# ── Neural value network ──────────────────────────────────────────────────────

class NeuralValue(nn.Module):
    """MLP value function: one-hot state -> scalar V(s)."""
    def __init__(self, n_states: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._n_states = n_states

    def _one_hot(self, s: int) -> torch.Tensor:
        x = torch.zeros(self._n_states); x[s] = 1.0; return x

    def forward(self, s: int) -> torch.Tensor:
        return self.net(self._one_hot(s)).squeeze(-1)


# ── Clipped surrogate loss ────────────────────────────────────────────────────

def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # (T,) current policy log probs
    log_probs_old: torch.Tensor,   # (T,) old policy log probs (detached)
    advantages:    torch.Tensor,   # (T,) advantage estimates
    epsilon:       float = 0.2,
) -> torch.Tensor:
    """
    L^CLIP(θ) = E_t[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]

    r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  — importance sampling ratio

    The clip prevents r_t from moving too far from 1, acting as a soft
    trust region without explicit KL constraint (unlike TRPO).
    """
    r_t  = torch.exp(log_probs_new - log_probs_old)          # importance ratio
    obj1 = r_t * advantages                                   # unclipped
    obj2 = torch.clamp(r_t, 1 - epsilon, 1 + epsilon) * advantages  # clipped
    return -torch.min(obj1, obj2).mean()                      # gradient ascent


# ── PPO training loop ─────────────────────────────────────────────────────────

def ppo(
    env,
    policy:       NeuralPolicy,
    value:        NeuralValue,
    n_iterations: int   = 200,
    n_steps:      int   = 512,    # steps collected per iteration
    n_epochs:     int   = 4,      # gradient epochs per iteration
    lr:           float = 3e-4,
    gamma:        float = 0.99,
    epsilon:      float = 0.2,
    vf_coef:      float = 0.5,
) -> Dict[str, List[float]]:
    """
    PPO with clipped surrogate. Collects n_steps of experience, then
    performs n_epochs of gradient updates on the collected batch.

    Returns history dict with keys: returns, pg_loss, v_loss, ratio_mean.
    """
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=lr
    )
    history: Dict[str, List[float]] = {
        "returns": [], "pg_loss": [], "v_loss": [], "ratio_mean": []
    }

    s = env.reset()
    ep_return, ep_returns = 0.0, []

    for it in range(n_iterations):
        # ── Collect rollout ───────────────────────────────────────────────
        states, actions, log_probs_old, rewards, dones, values_ = \
            [], [], [], [], [], []

        for _ in range(n_steps):
            with torch.no_grad():
                a, lp = policy.action_and_logprob(s)
                v      = value(s)
            s2, r, done = env.step(a)
            states.append(s); actions.append(a); log_probs_old.append(lp)
            rewards.append(r); dones.append(done); values_.append(v)
            ep_return += r
            s = env.reset() if done else s2
            if done:
                ep_returns.append(ep_return); ep_return = 0.0

        # ── Compute discounted returns ────────────────────────────────────
        G = torch.zeros(n_steps)
        g = 0.0
        for t in reversed(range(n_steps)):
            g    = rewards[t] + gamma * g * (1 - dones[t])
            G[t] = g

        log_probs_old_t = torch.stack(log_probs_old).detach()
        values_t        = torch.stack(values_).detach()
        advantages      = (G - values_t)
        advantages      = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO epochs ────────────────────────────────────────────────────
        for _ in range(n_epochs):
            lp_new = torch.stack([policy.log_prob(s, a)
                                   for s, a in zip(states, actions)])
            v_new  = torch.stack([value(s) for s in states])

            pg_loss = ppo_clip_loss(lp_new, log_probs_old_t, advantages, epsilon)
            v_loss  = F.mse_loss(v_new, G)
            loss    = pg_loss + vf_coef * v_loss

            opt.zero_grad(); loss.backward(); opt.step()

        r_mean = torch.exp(
            torch.stack([policy.log_prob(s, a) for s, a in zip(states, actions)]).detach()
            - log_probs_old_t
        ).mean().item()

        history["returns"].append(np.mean(ep_returns[-10:]) if ep_returns else 0.0)
        history["pg_loss"].append(float(pg_loss))
        history["v_loss"].append(float(v_loss))
        history["ratio_mean"].append(r_mean)

    return history
