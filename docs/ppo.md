# Proximal Policy Optimization — The Clipped Surrogate

**Companion to `notebooks/ppo/explore.ipynb`.** Continues `docs/policy_gradient.md`;
§7 there (the natural gradient and its KL trust region) is the conceptual parent of the clip.

*Drafted with the assistance of Claude (Anthropic).*

---

## 1. Off-policy reuse and importance sampling

REINFORCE (`policy_gradient.md` §5) is **on-policy**: each gradient step needs fresh samples
from the current $\pi_\theta$, which is sample-inefficient. To reuse a batch collected under an
older policy $\pi_{\theta_{\text{old}}}$, correct the expectation by importance sampling:

$$\mathbb E_{a\sim\pi_\theta}\big[A(s,a)\big]
 = \mathbb E_{a\sim\pi_{\theta_{\text{old}}}}\Big[\,\underbrace{\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}}_{\displaystyle r_t(\theta)}\,A(s,a)\Big].$$

The surrogate whose gradient at $\theta=\theta_{\text{old}}$ reproduces the policy gradient is
$L^{\text{IS}}(\theta)=\mathbb E_{\text{old}}[r_t(\theta)A_t]$, since $r_t(\theta_{\text{old}})=1$
and $\nabla_\theta r_t\big|_{\theta_{\text{old}}}=\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ — so
at the first step it *is* the policy gradient, but it lets us keep using the batch as $\theta$
moves away from $\theta_{\text{old}}$.

## 2. Why the ratio explodes

The ratio $r_t(\theta)=\pi_\theta/\pi_{\theta_{\text{old}}}$ is **unbounded above**. A
single-step surrogate $L_t=r_t A_t$ can blow up through two independent factors:

1. **Ratio explosion.** If an action was rare under the old policy
   ($\pi_{\text{old}}(a_t)$ small) but the new policy lifted it, $r_t\gg 1$. The IS
   correction is then extrapolating far outside the region where the old samples carry
   evidence.
2. **Advantage magnitude.** A large $|A_t|$ — possibly from one sparse, high-variance
   rollout — multiplies that ratio.

Their product means a single rare sample with a noisy advantage can dominate the batch
gradient and trigger an enormous, unreliable step. This is the disease §7 of the
policy-gradient notes diagnosed: nothing here constrains movement in *policy* space, only in
parameter space. The notebook's summary separates the two regimes — a **justified** big update
(high advantage on an action the old policy already sampled often) from a **spurious** one (a
rare action whose advantage estimate is sparse and high-variance).

## 3. The clipped surrogate

PPO bounds the multiplier without touching the signal. Clip the ratio to
$[1-\varepsilon,1+\varepsilon]$ and take the pessimistic (minimum) of the clipped and
unclipped surrogates:

$$\boxed{\ L^{\text{CLIP}}(\theta) = \mathbb E_t\Big[\min\!\big(r_t(\theta)\,A_t,\ \operatorname{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,A_t\big)\Big].\ }$$

The $\min$ makes $L^{\text{CLIP}}$ a **lower bound** on the unclipped surrogate, and the sign
of $A_t$ selects which clip binds:

- **$A_t>0$** (good action): the term grows with $r_t$, but the $\min$ caps the contribution
  once $r_t>1+\varepsilon$ — the gradient is zeroed beyond the cap, so there is no reward for
  pushing the action's probability up further once the policy has moved enough.
- **$A_t<0$** (bad action): the term grows as $r_t$ falls, capped once $r_t<1-\varepsilon$ —
  gradient zeroed below the cap, no reward for suppressing further.

Crucially the **advantage $A_t$ is left untouched**: PPO still responds at full strength to
genuine signal; only the off-policy multiplier $r_t$ is bounded, so a single step's
contribution is $\lesssim (1+\varepsilon)\,|A_t|$. Real high-advantage updates still happen —
they are just spread over more iterations instead of one giant leap. The threshold
$\varepsilon$ is a **trust-region radius**: small $\varepsilon$ is conservative and slow,
large $\varepsilon$ approaches unclipped importance sampling and risks instability. The
notebook sweeps $\varepsilon\in\{0.05,0.2,0.5\}$ to show the tradeoff.

## 4. Relation to TRPO and the natural gradient

PPO is a cheap, first-order stand-in for an explicit **KL trust region**. TRPO solves

$$\max_\theta\ \mathbb E_{\text{old}}\big[r_t(\theta)\,A_t\big]
 \quad\text{s.t.}\quad \mathbb E\big[\operatorname{KL}(\pi_{\theta_{\text{old}}}\,\|\,\pi_\theta)\big]\le\delta,$$

whose linear-quadratic solution is exactly the natural-gradient step
$\Delta\propto F^{-1}\nabla_\theta J$ of `policy_gradient.md` §7 — steepest ascent under the
Fisher metric, with $\delta$ the KL budget. TRPO enforces the constraint exactly (a
Fisher-vector-product conjugate-gradient solve); PPO replaces the hard KL ball with the
per-sample ratio clip, approximating "stay near $\pi_{\text{old}}$" using only first-order
quantities and no Fisher solve. The clip is to the KL trust region what a box constraint is to
an ellipsoidal one — coarser, but essentially free.
