import gymnasium as gym
import numpy as np
from models.q_learner import QL
import pandas as pd
import matplotlib.pyplot as plt

def train(n_episodes = 10000, is_slippery = False, path = "fl_weights.npy"):
  env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=is_slippery, render_mode=None)
  ql = QL(env, n_episodes)
  for i in range (n_episodes):
    ql.train_step(env, i)
  np.save(path, ql.q)
  rolling_rewards = pd.Series(ql.total_rewards).rolling(window=10).mean()
  plt.plot(rolling_rewards)
  plt.xlabel("Episode")
  plt.ylabel("Smoothed Total Reward")
  plt.title("Smoothed reward accumulation over episodes")
  plt.savefig("frozen_training.png")

def play(n = 1, render = None, is_slippery = False, path = "fl_weights.npy"):
  print("FrozenLake")
  env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=is_slippery, render_mode=render)
  ql = QL(env)
  ql.load_q(path)
  total_reward = 0
  for _ in range(n):
    terminated = False
    truncated = False
    state = env.reset()[0]
    while not(terminated or truncated):
      state, r, terminated, truncated, _i = env.step(ql.choose_action(state))
      total_reward += r
  print(f"{n} games played")
  print(f"Average reward per game: {total_reward/n}")
  env.close()

