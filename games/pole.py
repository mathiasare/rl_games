import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from models.qfa_learner import QFA
from models.ac_learner import AC

class PoleStrategy(Enum):
  QFA = 0,
  AC = 1

def train(n_episodes = 10000, strategy = PoleStrategy.QFA):
  env = gym.make("CartPole-v1", render_mode=None)
  match strategy:
    case PoleStrategy.QFA:
      model = QFA(env, n_episodes)
      path = "cart_weights_qfa.npy"
    case PoleStrategy.AC:
      model = AC(env, n_episodes)
      path = "cart_weights_ac.pth"
  for i in range(n_episodes):
    model.train_step(env, i)
  model.save_w(path)
  rolling_rewards = pd.Series(model.total_rewards).rolling(window=10).mean()
  plt.plot(rolling_rewards)
  plt.xlabel("Episode")
  plt.ylabel("Smoothed Total Reward")
  plt.title("Smoothed reward accumulation over episodes")
  plt.savefig("reward_plot.png")

def play(n = 1, strategy = PoleStrategy.QFA, render = None):
  print("CartPole-v1")
  env = gym.make("CartPole-v1", render_mode=render)
  match strategy:
    case PoleStrategy.QFA:
      model = QFA(env)
      path = "cart_weights_qfa.npy"
    case PoleStrategy.AC:
      model = AC(env, 0)
      path = "cart_weights_ac.pth"
  model.load_w(path)
  total_r = 0
  for _ in range(n):
    state = env.reset()[0]
    terminated = False
    truncated = False
    while not(terminated or truncated):
      state, r, terminated, truncated, _i = env.step(model.choose_action(state))
      total_r += r
  print(f"{n} games played")
  print(f"Average reward per game: {total_r/n}")
  env.close()




