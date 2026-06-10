"""
Retrain PPO with hardware-accurate friction.

Sysid finding: real Fc > 8.1 rad/s² (disc stuck at 3.6° with no motor).
Simulation default is Fc = 6.06 — too low by ~33%.

Tune FC_REAL below, run this script, then test on hardware with run_on_hardware.py.
"""

import sys, os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib
matplotlib.use('Agg')  # non-interactive — saves PNG without opening a window
from matplotlib import pyplot as plt

sys.path.insert(0, '.')
import gym_unbalanced_disk
from gym_unbalanced_disk.envs.UnbalancedDisk import UnbalancedDisk_sincos

# ── Tune this ─────────────────────────────────────────────────────────────────
FC_REAL      = 10.0    # hardware friction estimate — try 8, 10, 12
TOTAL_STEPS  = 300_000
MODEL_NAME   = f'ppo_swingup_fc{int(FC_REAL)}'
# ──────────────────────────────────────────────────────────────────────────────


class UnbalancedDisk_HW(UnbalancedDisk_sincos):
    """Simulation with hardware-accurate friction."""
    def __init__(self):
        super().__init__(umax=3.0, dt=0.025)
        self.Fc = FC_REAL
        self.reward_fun = lambda self: 0.5 * (1.0 - np.cos(self.th))


def make_env():
    return UnbalancedDisk_HW()


# ── Train ─────────────────────────────────────────────────────────────────────

print(f"Training with Fc = {FC_REAL}  (default = 6.06)")
print(f"Model will be saved as: {MODEL_NAME}.zip")

vec_env = make_vec_env(make_env, n_envs=4)
model = PPO(
    'MlpPolicy', vec_env,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.005,
    n_steps=1024,
    batch_size=64,
    verbose=1,
    seed=42,
)
model.learn(total_timesteps=TOTAL_STEPS)
model.save(MODEL_NAME)
print(f"Saved {MODEL_NAME}.zip")


# ── Evaluate in simulation ────────────────────────────────────────────────────

print("\nEvaluating in simulation...")
N_EPISODES = 5
episode_rewards = []

for ep in range(N_EPISODES):
    env = make_env()
    obs, _ = env.reset()
    obs_hist, act_hist, rew_hist = [], [], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_hist.append(obs.copy())
        act_hist.append(float(action))
        obs, r, terminated, truncated, _ = env.step(action)
        rew_hist.append(r)
        done = terminated or truncated

    total_r = sum(rew_hist)
    episode_rewards.append(total_r)
    print(f"  Episode {ep+1}: reward = {total_r:.1f} / {len(rew_hist)}")

print(f"\nMean reward: {np.mean(episode_rewards):.1f} / 300")


# ── Plot last episode ─────────────────────────────────────────────────────────

obs_arr = np.array(obs_hist)
th_deg  = np.degrees(np.arctan2(obs_arr[:, 0], obs_arr[:, 1]))
t       = np.arange(len(th_deg)) * 0.025

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f'PPO — Fc={FC_REAL}  |  reward={episode_rewards[-1]:.1f}/300')

axes[0].plot(t, th_deg, color='steelblue')
axes[0].axhline( 180, color='red',  linestyle='--', linewidth=0.8, label='target (180°)')
axes[0].axhline(-180, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_ylabel('θ (deg)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t, obs_arr[:, 2], color='darkorange')
axes[1].set_ylabel('ω (rad/s)')
axes[1].grid(True)

axes[2].plot(t, act_hist, color='green')
axes[2].axhline( 3.0, color='red', linestyle='--', linewidth=0.8)
axes[2].axhline(-3.0, color='red', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('u (V)')
axes[2].set_xlabel('time (s)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f'{MODEL_NAME}_eval.png', dpi=120)
plt.show()
