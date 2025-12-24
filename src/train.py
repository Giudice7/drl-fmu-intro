import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env import BuildingEnv
from settings import PROJECT_ROOT


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

with open("config.json", "r") as f:
    config = json.load(f)

env = BuildingEnv(config)
env = Monitor(env, log_dir)

n_steps = 8928
n_episodes = 40

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=n_steps, policy_kwargs = dict(
    net_arch=dict(pi=[128, 64, 16], vf=[128, 64, 16])
))

print("Starting training...")
model.learn(total_timesteps=n_steps * n_episodes)
print("Training finished.")

model.save("ppo_simplehouse_model")
env.close()

data = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)

x = data.index
y = data['r']

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.3, label='Raw Reward')

if len(y) > 1:
    window_size = max(1, int(len(y) * 0.1))
    y_smoothed = data['r'].rolling(window=window_size).mean()
    plt.plot(x, y_smoothed, color='orange', label='Moving Average')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward per Episode')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'training_reward.png'))
plt.show()
