import gymnasium as gym
import numpy as np

class QL:
  def __init__(self, env: gym.Env, n_episodes = 1) -> None:
    self.alpha = 0.9 # learning rate
    self.gamma = 0.9 # discount factor
    self.epsilon = 1
    self.epsilon_decay = 0.0001
    self.q = np.zeros((env.observation_space.n, env.action_space.n))
    self.rng = np.random.default_rng()
    self.total_rewards = np.zeros(n_episodes, dtype=np.int16)

  def q_update(self, state, action, reward, new_state):
    self.q[state,action] = self.q[state,action] + self.alpha * (
      reward + self.gamma * np.max(self.q[new_state,:]) - self.q[state,action]
    )

  def load_q(self, path):
    self.q = np.load(path)

  def train_step(self, env: gym.Env, episode):
    state = env.reset()[0]
    episode_over = False
    reward = 0
    while not (episode_over):
      action = self.choose_action_train(env, state)
      new_state, reward, terminated, truncated, _info = env.step(action)
      self.q_update(state, action, reward, new_state)
      episode_over = terminated or truncated
      state = new_state
      self.global_step += 1
    self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
    if (reward == 1):
      self.total_rewards[episode] = 1

  def choose_action_train(self, env: gym.Env, state):
    if self.rng.random() < self.epsilon:
      return env.action_space.sample()
    else:
      return np.argmax(self.q[state, :])

  def choose_action(self, state):
    return np.argmax(self.q[state, :])