import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
from models.dq_learner import DQL
from models.qfa_learner import QFA

def train(n = 1000, path="lunar_lander.pth"):
  env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = None)
  dql = DQL(n_episodes=n, env=env)
  pbar = trange(n, desc="Training", unit="episode")
  best_reward = -100_000
  for i in pbar:
    reward, epsilon = dql.train_step(env, i)
    if (reward > best_reward):
      best_reward = reward
    pbar.set_postfix({"reward": f"{reward:.1f}", "best_reward": f"{best_reward:.1f}", "epsilon": f"{epsilon:.3f}"})
  dql.save_nn(path)
  rolling_rewards = pd.Series(dql.total_rewards).rolling(window=100).mean()
  plt.plot(rolling_rewards)
  plt.xlabel("Episode")
  plt.ylabel("Smoothed Total Reward")
  plt.title("Smoothed reward accumulation over episodes")
  plt.savefig("reward_plot_lunar.png")


def play(n = 100, render = None, path="lunar_lander.pth"):
  print("LunarLander-v3")
  env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = render)
  dql = DQL(n_episodes=n, env=env)
  dql.load_nn(path)
  total_r = 0
  for _ in range(n):
    state = env.reset()[0]
    terminated = False
    truncated = False
    while not(terminated or truncated):
      state, r, terminated, truncated, _i = env.step(dql.choose_action(state))
      total_r += r
  print(f"{n} games played")
  print(f"Average reward per game: {total_r/n}")
  env.close()