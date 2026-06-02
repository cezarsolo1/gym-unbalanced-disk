# PPO Full Numerical Walkthrough
### One complete training step on the Unbalanced Disc

---

## Setup

The disc starts at **45°**, spinning upward at 1.2 rad/s.

```
         |  ← top (180°)
         |
    ●────┤  ← disc at 45°, moving upward
         |
         0  ← bottom (start)
```

---

## Step 1 — Observation

The network never sees the raw angle θ. It sees:

| Input | Formula | Value |
|-------|---------|-------|
| sin θ | sin(45°) | **0.7069** |
| cos θ | cos(45°) | **0.7073** |
| ω     | angular velocity | **1.1998 rad/s** |

**Why sin/cos instead of θ?**
At 179° and -179° the disc is almost in the same place, but the raw numbers look totally different. sin/cos wraps around smoothly so the network is never confused.

---

## Step 2 — Actor Network (what voltage to apply?)

```
[0.7069, 0.7073, 1.1998]
        ↓
   Linear layer (3 → 64)
   + tanh activation
        ↓
   64 hidden values  (e.g. -0.168, -0.998, -0.454, 0.351, 1.0 ...)
        ↓
   Linear layer (64 → 1)
        ↓
      μ = 1.2005 V   ← the mean voltage the network recommends
```

The actor doesn't output a single fixed voltage. It outputs a **Gaussian distribution**:

$$\pi(u \mid x) = \mathcal{N}(\mu,\ \sigma^2)$$

| Parameter | Value | Meaning |
|-----------|-------|---------|
| μ (mean) | **1.2005 V** | centre of the distribution — best guess |
| log σ | **-0.9453** | learnable, stored as log for stability |
| σ (std) | **0.3886 V** | spread — how much it still explores |

During training we **sample** from this distribution:

$$u \sim \mathcal{N}(1.2005,\ 0.3886^2) \rightarrow u = 0.9151 \text{ V}$$

The log-probability of having chosen this action:

$$\log \pi(u \mid x) = -\frac{(0.9151 - 1.2005)^2}{2 \times 0.3886^2} - \log(0.3886\sqrt{2\pi}) = -0.2435$$

This number will be used later in the actor loss.

---

## Step 3 — Critic Network (how good is this state?)

A separate branch of the network takes the same observation and outputs **one number** — its estimate of how much total future reward to expect from here:

```
[0.7069, 0.7073, 1.1998]
        ↓
   Linear layer (3 → 64)
   + tanh activation
        ↓
   64 hidden values  (e.g. -0.9995, 1.0, 0.9999 ...)
        ↓
   Linear layer (64 → 1)
        ↓
      V(x_now) = 90.04
```

**V(x) = 90.04** means: *"from this state, I expect to collect about 90 more reward points over the rest of the episode."*

---

## Step 4 — Environment Step

The sampled voltage **0.9151 V** gets sent to the disc simulator for 0.025 s.

The physics equations run:

$$\dot{\omega} = -\omega_0^2 \sin\theta - \gamma\omega - F_c\,\text{sign}(\omega) + K_u \cdot u$$

| Before | After |
|--------|-------|
| θ = 45.00° | θ = 45.47° |
| ω = 1.1998 rad/s | ω = -0.4585 rad/s |

The disc moved slightly upward. The reward formula fires:

$$r = \frac{1 - \cos(45.47°)}{2} = \frac{1 - 0.7012}{2} = 0.1493$$

---

## Step 5 — Critic evaluates the next state

The critic now looks at the new observation and estimates future reward from there:

$$V(x_{next}) = 90.26$$

---

## Step 6 — Advantage Calculation

$$A = r + \gamma \cdot V(x_{next}) - V(x_{now})$$

$$A = 0.1493 + 0.99 \times 90.26 - 90.04$$

$$A = 0.1493 + 89.36 - 90.04 = \mathbf{-0.5265}$$

**Interpretation:** The advantage is **negative**.

The critic expected 90.04 from this state. The actual outcome (reward + discounted next value) was only 89.51. The action was **worse than expected** — the disc slowed down and ω flipped sign instead of continuing to build momentum upward.

---

## Step 7 — Three Losses

### Critic loss — how wrong was the value estimate?

$$\mathcal{L}_{critic} = A^2 = (-0.5265)^2 = 0.2772$$

The critic was off by 0.5265. Backprop will push V(x_now) from **90.04 → 89.51** (toward what actually happened).

---

### Actor loss — should this action become more or less likely?

$$\mathcal{L}_{actor} = -A^{stop} \cdot \log\pi(u \mid x) = -(-0.5265) \times (-0.2435) = -0.1282$$

The $A^{stop}$ means the advantage is treated as a **fixed constant** here — gradients do not flow back through it into the critic. Only the actor weights move.

Since A is negative, this loss **increases** the actor weights in the direction that makes u = 0.9151 V **less likely** next time. The mean μ will shift slightly away from 0.9151 V.

---

### Entropy bonus — keep exploring

$$\mathcal{L}_{entropy} = -\mathcal{S}(\pi) = -\log(\sigma\sqrt{2\pi e}) = -0.4737$$

A penalty for letting σ collapse to zero too early. Keeps the policy spread out so it keeps trying different voltages.

---

### Total loss

$$\mathcal{L} = \mathcal{L}_{critic} + \alpha_a \cdot \mathcal{L}_{actor} + \alpha_S \cdot \mathcal{L}_{entropy}$$

$$\mathcal{L} = 0.2772 + 1.0 \times (-0.1282) + 0.005 \times (-0.4737)$$

$$\mathcal{L} = 0.2772 - 0.1282 - 0.0024 = \mathbf{0.1467}$$

---

## Step 8 — Weight Update

PyTorch runs backpropagation through the total loss and computes the gradient of every weight in both networks. Then:

$$\theta \leftarrow \theta - \underbrace{3 \times 10^{-4}}_{\text{lr}} \cdot \nabla_\theta \mathcal{L}$$

| Network | Before | After |
|---------|--------|-------|
| Actor mean output | 1.2005 V | shifts slightly away from 0.9151 V |
| Critic value output | 90.04 | shifts toward 89.51 |
| Actor σ | 0.3886 | kept from collapsing by entropy term |

The shift per step is tiny (lr = 0.0003). This same process repeats across **thousands of steps**, gradually shaping the weights until the policy reliably swings the disc to the top.

---

## PPO Clip (the safety check)

Before any of the above losses are applied, PPO checks:

$$r_t = \frac{\pi_{new}(u \mid x)}{\pi_{old}(u \mid x)}$$

If this ratio is outside $[1 - \varepsilon,\ 1 + \varepsilon] = [0.8,\ 1.2]$, the gradient is **zeroed out** — the update is blocked. This prevents one bad step from destroying weeks of learned behaviour.

---

## Full Picture

```
  obs = [0.707, 0.707, 1.200]
         │
         ├──► Actor ──► μ = 1.20 V, σ = 0.39 ──► sample u = 0.92 V
         │                                                │
         └──► Critic ──► V(x_now) = 90.04               │
                                                          ▼
                                              Environment (physics)
                                                          │
                                              r = 0.1493, x_next
                                                          │
                                              Critic ──► V(x_next) = 90.26
                                                          │
                                              A = 0.15 + 89.36 - 90.04 = -0.53
                                                          │
                              ┌───────────────────────────┤
                              │                           │
                         L_critic = 0.28          L_actor = -0.13
                              │                           │
                              └──────── L_total = 0.15 ───┘
                                                │
                                         backprop + step
                                                │
                                    weights updated (lr = 3e-4)
```
