# Policy Gradient Methods — Derivations

**Companion to `notebooks/policy_gradient/explore.ipynb`.** That notebook is a
direct transcription of the equations below into runnable code on a 5×5
GridWorld; each notebook section cites a section number here.

<small>*Drafted with the assistance of Claude (Anthropic).*</small>

---

## 1. The MDP and the objective

A finite Markov decision process is the tuple $(\mathcal S, \mathcal A, P, r, \gamma, \rho_0)$:
states $\mathcal S$, actions $\mathcal A$, transition kernel $P(s'\mid s,a)$, reward
$r(s,a)$, discount $\gamma\in[0,1)$, and initial-state distribution $\rho_0$. A
stochastic policy $\pi_\theta(a\mid s)$ with parameters $\theta$ induces a distribution
over trajectories $\tau=(s_0,a_0,s_1,a_1,\dots)$,

$$p_\theta(\tau) = \rho_0(s_0)\prod_{t\ge 0}\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t).$$

The discounted return is $R(\tau)=\sum_{t\ge 0}\gamma^t r(s_t,a_t)$, and the objective is

$$J(\theta) = \mathbb E_{\tau\sim p_\theta}\big[R(\tau)\big].$$

We want $\nabla_\theta J$ in order to ascend it. The obstacle is that the expectation is
taken over $p_\theta$, which itself depends on $\theta$, so $\nabla_\theta$ cannot be
moved inside the integral naïvely.

## 2. The softmax policy

The notebook uses a **tabular softmax** policy: one logit row $\theta_{s,:}\in\mathbb R^{|\mathcal A|}$
per state, with

$$\pi_\theta(a\mid s) = \operatorname{softmax}(\theta_{s,:})_a = \frac{e^{\theta_{s,a}}}{\sum_{a'}e^{\theta_{s,a'}}}.$$

This is the minimal differentiable policy class — no function approximation, every state
independent — so the only thing learned is the per-state action preference. It keeps the
geometry of §7 transparent.

## 3. The score function

The quantity that makes the gradient tractable is the **score**
$\nabla_\theta\log\pi_\theta(a\mid s)$. Writing
$\log\pi_\theta(a\mid s)=\theta_{s,a}-\log\sum_{a'}e^{\theta_{s,a'}}$ and differentiating
with respect to an arbitrary logit $\theta_{s,a''}$ at the same state,

$$\frac{\partial}{\partial\theta_{s,a''}}\log\pi_\theta(a\mid s)
 = \mathbf 1[a''=a] - \frac{e^{\theta_{s,a''}}}{\sum_{a'}e^{\theta_{s,a'}}}
 = \mathbf 1[a''=a] - \pi_\theta(a''\mid s).$$

The score is $+1$ on the taken action minus the policy's probability vector — the standard
log-softmax gradient. (At a different state $\tilde s\neq s$ the derivative vanishes, since
$\theta_{\tilde s,:}$ does not enter $\pi_\theta(a\mid s)$.) The notebook checks this against
`log_prob.backward()`.

One property is used repeatedly: the score has **zero mean** under the policy,

$$\mathbb E_{a\sim\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(a\mid s)\big]
 = \sum_a \pi_\theta(a\mid s)\,\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
 = \nabla_\theta\sum_a\pi_\theta(a\mid s) = \nabla_\theta 1 = 0. \tag{$\star$}$$

This is the discrete analogue of $\mathbb E[\nabla\log p]=0$ for any normalised family, and it
is what makes baselines free (§6) and the Fisher information well-defined (§7).

## 4. The policy gradient theorem

Apply the **log-derivative identity** $\nabla_\theta p_\theta=p_\theta\nabla_\theta\log p_\theta$:

$$\nabla_\theta J = \nabla_\theta\!\int p_\theta(\tau)R(\tau)\,d\tau
 = \int p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)\,R(\tau)\,d\tau
 = \mathbb E_\tau\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big].$$

Because $\rho_0$ and $P$ do not depend on $\theta$, taking the log of $p_\theta(\tau)$ removes
them, leaving only policy terms:

$$\nabla_\theta\log p_\theta(\tau) = \sum_{t\ge 0}\nabla_\theta\log\pi_\theta(a_t\mid s_t),
\qquad
\nabla_\theta J = \mathbb E_\tau\Big[\Big(\textstyle\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big)R(\tau)\Big].$$

**Reward-to-go (causality).** A reward $r_{t'}$ earned *before* action $a_t$ cannot depend on
$a_t$, so in expectation its cross term vanishes. For $t'<t$, condition on the history up to
$t$ and use $(\star)$:

$$\mathbb E\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\gamma^{t'}r_{t'}\big]
 = \mathbb E\Big[\gamma^{t'}r_{t'}\,\underbrace{\mathbb E_{a_t\sim\pi}[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\mid s_t]}_{=\,0}\Big]=0.$$

Dropping these terms gives the **reward-to-go** form with
$G_t=\sum_{t'\ge t}\gamma^{t'-t}r_{t'}$:

$$\boxed{\ \nabla_\theta J = \mathbb E_\tau\Big[\textstyle\sum_{t\ge 0}\gamma^{t}\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big].\ }$$

(The notebook folds $\gamma^t$ into the return convention; at $\gamma=0.99$ and short
episodes the difference from the common undiscounted-weighting estimator is small.)

## 5. REINFORCE

REINFORCE is the Monte-Carlo estimator of the boxed gradient: roll out $N$ trajectories
under $\pi_\theta$, form

$$\widehat{\nabla_\theta J} = \frac1N\sum_{i=1}^N\sum_t \nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)\,G_t^i,$$

and ascend, $\theta\leftarrow\theta+\alpha\,\widehat{\nabla_\theta J}$. The estimator is
**unbiased** but **high-variance**: $G_t$ aggregates all downstream stochasticity
(dynamics, policy, reward), and that variance enters the gradient linearly. Reducing it
without introducing bias is the job of the baseline.

## 6. Baselines and the advantage

Subtract a **state-dependent baseline** $b(s_t)$ from the return. By $(\star)$ it contributes
nothing in expectation,

$$\mathbb E\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big]
 = \mathbb E\Big[b(s_t)\,\underbrace{\mathbb E_{a_t\sim\pi}[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\mid s_t]}_{=\,0}\Big]=0,$$

so the estimator stays unbiased for *any* $b$, while its variance can drop sharply. Taking
$b(s)=V(s)$, the on-policy value, recentres each score by the **advantage**

$$\hat A_t = G_t - V(s_t),\qquad A(s,a) = Q(s,a) - V(s),$$

which asks "was this action better than the policy's average from $s_t$?" Geometrically, the
score $\nabla_\theta\log\pi(a\mid s)$ is a fixed star of directions in logit space (push the
taken action up, all others down); the advantage is the signed length that scales it.
Centring by $V(s)$ removes the component common to all actions at $s$ — the part that, by
$(\star)$, exerts zero net force but inflates variance.

The notebook fits $V$ by regression, $V(s_t)\leftarrow V(s_t)+\alpha_V\,(G_t-V(s_t))$, builds
the table $A(s,a)$ by Monte-Carlo rollouts that fix the first action, and applies

$$\theta \leftarrow \theta + \alpha\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t.$$

## 7. Why Euclidean gradient ascent is suboptimal — the natural gradient

Ordinary ascent $\theta\leftarrow\theta+\alpha\nabla_\theta J$ takes the fixed-length
**Euclidean** step that most increases $J$. But $\theta$ is merely a coordinate chart; what
we care about is movement of the *distribution* $\pi_\theta$, and equal steps in $\theta$
produce unequal changes in $\pi_\theta$. The natural notion of distance between policies is
the KL divergence, and to second order it defines a Riemannian metric — the **Fisher
information**.

Expand $\operatorname{KL}\!\big(\pi_\theta\,\|\,\pi_{\theta+\Delta}\big)$ for small $\Delta$.
The zeroth and first orders vanish (KL is minimised at $\Delta=0$ with value $0$), leaving

$$\operatorname{KL}\!\big(\pi_\theta\,\|\,\pi_{\theta+\Delta}\big)
 = \tfrac12\,\Delta^\top F(\theta)\,\Delta + O(\|\Delta\|^3),
\qquad
F(\theta) = \mathbb E_{a\sim\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta\,\nabla_\theta\log\pi_\theta^\top\big].$$

$F$ is the Fisher information — the covariance of the score, and equivalently
$F=\mathbb E[-\nabla_\theta^2\log\pi_\theta]$; the two forms coincide precisely because of
$(\star)$. Now pose steepest ascent in the *distribution* metric: maximise the linearised
objective under a fixed KL budget,

$$\max_{\Delta}\ \nabla_\theta J^\top \Delta
 \quad\text{s.t.}\quad \tfrac12\,\Delta^\top F\,\Delta \le \varepsilon.$$

A Lagrange multiplier gives $\Delta \propto F^{-1}\nabla_\theta J$ — the **natural gradient**

$$\widetilde{\nabla}_\theta J = F^{-1}\,\nabla_\theta J.$$

It is the steepest-ascent direction with respect to the Fisher–Rao metric on the policy
manifold, and it is invariant to smooth reparameterisations of $\theta$ (the $F^{-1}$ cancels
the Jacobians a coordinate change introduces) — unlike the Euclidean gradient, whose
direction depends on how you happened to parameterise.

The notebook exposes the gap without ever forming $F$: from one policy it takes two steps of
**equal Euclidean length** in different directions and measures the resulting KL from the
start. The KLs differ — direct evidence that $F$ is anisotropic, so Euclidean length is the
wrong yardstick. Controlling the *policy-space* step instead of the parameter-space step is
exactly what TRPO and PPO do; see `docs/ppo.md`.
