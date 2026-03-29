import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, ReLU, Sequential
from collections import deque


class DQL:
  def __init__(self, n_episodes: int, env: gym.Env, epsilon = 1, epsilon_decay = 0.001, min_epsilon = 0.08, batch_size: int = 64, buffer_size: int = 100_000, gamma: float = 0.99, target_update_freq = 1000, device: str = "mps") -> None:
    self.state_dim = env.observation_space.shape[0]   # 8
    self.action_dim = env.action_space.n
    self.device = torch.device(device)
    self.q_online = QNetwork(self.state_dim, self.action_dim).to(device)
    self.q_target = QNetwork(self.state_dim, self.action_dim).to(device)
    self.q_target.load_state_dict(self.q_online.state_dict())
    self.q_target.eval()
    self.batch_size = batch_size
    self.gamma = gamma
    self.replay = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)
    self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=1e-3)
    self.loss_fn = torch.nn.MSELoss()
    self.total_rewards = np.zeros(n_episodes, dtype=np.int64)
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.min_epsilon = min_epsilon
    self.rng = np.random.default_rng()
    self.global_step = 0
    self.target_update_freq = target_update_freq

  def train_step(self, env: gym.Env, episode: int):
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
      action = self.choose_action_train(state, env)
      next_state, reward, terminated, truncated, _ = env.step(action)
      self.global_step += 1
      if self.global_step % self.target_update_freq == 0:
        self.q_target.load_state_dict(self.q_online.state_dict())
      done = terminated or truncated

      self.replay.push(state, action, reward, next_state, done)
      state = next_state
      total_reward += reward

      if (len(self.replay) < self.batch_size):
        continue
      states, actions, rewards, next_states, dones = self.replay.sample()
      states = torch.tensor(states, dtype=torch.float32).to(self.device)
      actions = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1)
      rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
      next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
      dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

      q_values = self.q_online(states).gather(1, actions)

      with torch.no_grad():
        next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.q_target(next_states).gather(1, next_actions)
        targets = rewards + self.gamma * (1 - dones) * next_q_values

      loss = self.loss_fn(q_values, targets)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
    self.total_rewards[episode] = total_reward
    return total_reward, self.epsilon

  def choose_action_train(self, state, env: gym.Env):
      if self.rng.random() < self.epsilon:
        return env.action_space.sample()
      state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      with torch.no_grad():
          q_values = self.q_online(state_t)
      return q_values.argmax(dim=1).item()

  def choose_action(self, state):
      state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      with torch.no_grad():
          q_values = self.q_online(state_t)
      return q_values.argmax(dim=1).item()

  def save_nn(self, path = "q_online.pth"):
    torch.save(self.q_online.state_dict(), path)

  def load_nn(self, path = "q_online.pth"):
    self.q_online.load_state_dict(torch.load(path))
    self.q_online.eval()

class QNetwork(torch.nn.Module):
  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.nn = Sequential(
      Linear(input_dim, 128),
      ReLU(),
      Linear(128, 128),
      ReLU(),
      Linear(128, output_dim)
    )

  def forward(self, x):
    return self.nn(x)

class ReplayBuffer:
  def __init__(self, batch_size: int, buffer_size: int) -> None:
    self.batch_size = batch_size
    self.buffer = deque(maxlen=buffer_size)

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self):
    indices = np.random.randint(0, len(self.buffer), size=self.batch_size)
    batch = [self.buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
    return states, actions, rewards, next_states, dones
  def __len__(self):
      return len(self.buffer)