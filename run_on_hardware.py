import sys
import os
import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import gym_unbalanced_disk

# Connects to the physical disc via USB — make sure the cable is plugged in
env = gym_unbalanced_disk.UnbalancedDisk_exp_sincos()

model = PPO.load(os.path.join(REPO_ROOT, 'ppo_swingup'))
print('Model loaded. Ctrl-C to stop.')
print('Waiting for disc to settle before first episode...')

obs, _ = env.reset()
total_reward = 0.0
step = 0

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        env.render()
        if terminated or truncated:
            th_deg = np.degrees(np.arctan2(obs[0], obs[1]))
            print(f'Episode done — steps: {step}, reward: {total_reward:.1f}/300, final θ: {th_deg:.1f}°')
            total_reward = 0.0
            step = 0
            print('Waiting for disc to settle...')
            obs, _ = env.reset()
except KeyboardInterrupt:
    print('Stopped.')
finally:
    env.close()
