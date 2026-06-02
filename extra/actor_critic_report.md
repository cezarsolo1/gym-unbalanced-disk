# 5SC28 Design Assignment — Actor-Critic Policy Learning

## Task

Learn a controller that swings the unbalanced disc from the **bottom position** (θ = 0°) to the **upright position** (θ = 180°) and keeps it stable.

The disc is an inverted pendulum driven by a DC motor. The only input is a voltage u ∈ [−3, +3] V. The system dynamics are:

$$\dot{\theta} = \omega, \qquad \dot{\omega} = -\omega_0^2\sin\theta - \gamma\omega - F_c\,\text{sign}(\omega) + K_u u$$

---

## Method: Actor-Critic Reinforcement Learning

Reinforcement learning frames the problem as an agent interacting with an environment. At each timestep the agent observes the state, takes an action, and receives a reward. The goal is to maximise the total reward.

**Actor-Critic** maintains two components simultaneously:

| Component | Role | Update rule |
|-----------|------|--------|
| **Actor** $\pi_\theta(u\|x)$ | Decides what action to take | Gradient ascent on expected reward |
| **Critic** $V_\eta(x)$ | Estimates how good the current state is | Minimise TD error |

The key quantity is the **advantage** — how much better or worse the action was than expected:

$$A_t = \underbrace{r_{t+1} + \gamma V_\eta(x_{t+1})}_{\text{what actually happened}} - \underbrace{V_\eta(x_t)}_{\text{what was expected}}$$

- $A_t > 0$: action was better than expected → increase its probability  
- $A_t < 0$: action was worse than expected → decrease its probability

The full training loss per mini-batch is:

$$\mathcal{L} = \underbrace{A_t^2}_{\text{train critic}} + \alpha_a \underbrace{(-A_t^{\text{sg}} \log \pi_\theta(u_t|x_t))}_{\text{train actor}} + \alpha_S \underbrace{(-\mathcal{S}(\pi_\theta))}_{\text{entropy bonus}}$$

where $\cdot^{\text{sg}}$ means stop-gradient (the advantage is treated as a constant when updating the actor), and the entropy term $\mathcal{S}$ prevents the policy from becoming deterministic too early.

---

## A2C Implementation from Scratch

To demonstrate understanding of the algorithm, we first implemented **A2C (Advantage Actor-Critic)** manually.

### Continuous Gaussian Policy

Since the action space is continuous (voltage), the actor outputs a Gaussian distribution instead of a softmax:

$$\pi_\theta(u|x) = \mathcal{N}(\mu_\theta(x),\, \sigma^2)$$

- $\mu_\theta(x)$: a neural network bounded by $\tanh \times u_{\max}$ so $\mu \in [-3, 3]$ V  
- $\sigma$: a single learnable scalar (same for all states, clamped to prevent explosion)

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=64, umax=3.0):
        super().__init__()
        # Critic: V(x)
        self.critic_l1 = nn.Linear(obs_dim, hidden_size)
        self.critic_l2 = nn.Linear(hidden_size, 1)
        # Actor: mu(x)
        self.actor_l1  = nn.Linear(obs_dim, hidden_size)
        self.actor_mu  = nn.Linear(hidden_size, 1)
        # Learnable log-std (clamped to [-2, 0.5])
        self.log_sigma = nn.Parameter(torch.tensor([0.0]))

    def critic(self, state):
        h = torch.tanh(self.critic_l1(state))
        return self.critic_l2(h)[:, 0]

    def actor(self, state):
        h     = torch.tanh(self.actor_l1(state))
        mu    = self.umax * torch.tanh(self.actor_mu(h)[:, 0])
        sigma = self.log_sigma.clamp(-2, 0.5).exp().expand_as(mu)
        return mu, sigma
```

### Training Loop

```
for each iteration:
    1. Collect N_rollout steps using the current stochastic policy
    2. For each mini-batch:
        - Compute advantage:  A = r + gamma*V(next) - V(now)
        - Normalise advantage (zero mean, unit std)
        - Actor loss:   L_pg  = -(A_detached * log_pi).mean()
        - Critic loss:  L_vf  = (A^2).mean()
        - Entropy loss: L_ent = -entropy.mean()
        - Total loss:   L = L_vf + alpha_actor*L_pg + alpha_entropy*L_ent
        - Backpropagate and update
    3. Evaluate deterministic policy and save best checkpoint
```

### Why A2C Struggled on this Task

The custom A2C implementation failed to learn the swing-up because:

1. **Sparse reward**: the cosine reward is near zero for most of the episode (disc near bottom), giving no useful gradient signal
2. **Sigma collapse**: with near-zero advantages, the entropy term dominated and drove sigma to extreme values
3. **No update constraint**: a single bad update could destroy the policy irreversibly

These are the exact limitations that PPO was designed to fix.

---

## Algorithm: PPO (Proximal Policy Optimisation)

We use **PPO** via stable-baselines3 for the final trained policy. PPO is the industry-standard actor-critic algorithm — it uses the same A2C framework but adds one key improvement: a **clip** that limits how much the policy can change in a single update:

$$L^{CLIP} = \mathbb{E}\left[\min\left(r_t A_t,\;\text{clip}(r_t,\,1-\varepsilon,\,1+\varepsilon)\,A_t\right)\right]$$

where $r_t = \pi_\theta(u|x) / \pi_{\theta_\text{old}}(u|x)$ is the ratio of new to old policy probability.

| | A2C (custom) | PPO (stable-baselines3) |
|---|---|---|
| Update constraint | None | Clip ratio to $[1-\varepsilon, 1+\varepsilon]$ |
| Data reuse | Single pass | Multiple epochs (safe to reuse) |
| Stability | Low | High |
| Implementation | From scratch | Pre-built library |

> **Note**: stable-baselines3 provides a well-tested, optimised implementation of PPO. We configure the hyperparameters and reward function; the algorithm internals are provided by the library. The custom A2C implementation above demonstrates understanding of the underlying theory.

---

## State and Action Space

| | Description |
|---|---|
| **Observation** | $[\sin\theta,\;\cos\theta,\;\omega]$ — sin/cos avoids angle wrap-around discontinuities |
| **Action** | Voltage $u \in [-3, +3]$ V (continuous) |
| **Episode length** | 300 steps × 0.025 s = **7.5 seconds** |

---

## Reward Function

We use a dense cosine reward that provides a gradient signal from every state:

$$r = \frac{1 - \cos\theta}{2}$$

| Position | Reward |
|----------|--------|
| Bottom (θ = 0°) | 0 |
| Side (θ = 90°) | 0.5 |
| Top (θ = 180°) | 1.0 |

The original environment reward (a narrow Gaussian at θ = 180°) was near zero everywhere except exactly at the top, making it impossible to learn from. The cosine reward provides gradient signal from the bottom up.

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (stable-baselines3) |
| Parallel environments | 4 |
| Total timesteps | 300,000 |
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | 0.99 |
| Entropy coefficient | 0.005 |

---

## Results

The PPO agent achieves a mean episode reward of **~281/300 (94%)**, successfully swinging the disc from the bottom to near the upright position and stabilising it for the remainder of the 7.5 second episode. The policy converged within ~150,000 timesteps.

---

## Advantages and Disadvantages

**Advantages of actor-critic / PPO:**
- Works directly from raw sensor data — no model of the system required
- Handles continuous action spaces naturally via Gaussian policy
- PPO is stable and robust to hyperparameter choices

**Disadvantages:**
- Requires many environment interactions (~300k steps)
- Reward shaping is critical — sparse rewards fail to train
- Policy behaviour outside the training distribution is not guaranteed
