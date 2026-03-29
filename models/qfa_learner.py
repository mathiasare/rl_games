import numpy as np
import gymnasium as gym

class QFA:
  def __init__(self, env: gym.Env, n_episodes = 0, known_vals = [2.4, 3.0, 0.21, 3.0]) -> None:
    self.alpha = 0.05
    self.gamma = 0.95 # discount factor
    self.epsilon = 1
    self.epsilon_decay = 0.001
    self.rng = np.random.default_rng()
    self.w = np.zeros((env.observation_space.shape[0], env.action_space.n), dtype=np.float64)
    self.total_rewards = np.zeros(n_episodes)
    self.known_max_values = np.array(known_vals, dtype=np.float32)
    self.n_actions = env.action_space.n

  def train_step(self, env, episode):
    state = env.reset()[0]
    state = self.normalize_state(state)
    episode_over = False
    reward = 0
    while not (episode_over):
      action = self.choose_action_train(env, state)
      new_state, reward, terminated, truncated, _info = env.step(action)
      new_state = self.normalize_state(new_state)
      self.q_update(state, action, reward, new_state)
      episode_over = terminated or truncated
      state = new_state
      self.total_rewards[episode] += reward
    self.epsilon = max(self.epsilon * 0.995, 0.05)

  def q(self, state, action):
    state = np.array(state)
    return self.w[:, action] @ state

  def best_q_action(self, state):
    return np.argmax(np.array([self.q(state, action) for action in range(self.n_actions)]))

  def best_q_value(self, state):
    return np.max(np.array([self.q(state, action) for action in range(self.n_actions)]))

  def normalize_state(self, state):
      state = np.array(state, dtype=np.float32)
      normed = state / self.known_max_values
      return np.clip(normed, -1, 1)

  def q_update(self, state, action, reward, new_state):
    qsa = self.q(state, action)
    target = reward + self.gamma * self.best_q_value(new_state)
    error = target - qsa
    self.w[:, action] = self.w[:, action] + self.alpha * error * np.array(state)

  def save_w(self, path):
    np.save(path, self.w)

  def load_w(self, path):
    self.w = np.load(path)

  def choose_action_train(self, env: gym.Env, state):
    if self.rng.random() < self.epsilon:
      return env.action_space.sample()
    else:
      return self.best_q_action(state)

  def choose_action(self, state):
    state = self.normalize_state(state)
    return self.best_q_action(state)
