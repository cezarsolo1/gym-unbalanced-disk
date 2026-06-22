import gym_unbalanced_disk, time

env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.)

obs, info = env.reset()
try:
    for i in range(500):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(1/24)
finally:
    env.close()
