import sys, time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.insert(0, '/Users/cezar/Documents/Documents_mac/University/TUe/Q6/gym-unbalanced-disk')
import gym_unbalanced_disk

env = gym_unbalanced_disk.UnbalancedDisk_sincos()
env.unwrapped.reward_fun = lambda self: 0.5 * (1.0 - np.cos(self.th))

model = PPO.load('ppo_swingup')
print('Model loaded. Close the pygame window to exit.')

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
        time.sleep(env.unwrapped.dt)
        if terminated or truncated:
            th_deg = np.degrees(np.arctan2(obs[0], obs[1]))
            print(f'Episode done — steps: {step}, total reward: {total_reward:.1f}/300, final θ: {th_deg:.1f}°')
            total_reward = 0.0
            step = 0
            obs, _ = env.reset()
finally:
    env.close()
