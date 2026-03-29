import gymnasium as gym
import torch
import numpy as np
from torch.nn import Sequential, ReLU, Linear, Softmax, Module

class AC:
  def __init__(self, env: gym.Env, n_episodes: int, actor_learning_rate = 5e-4, critic_learning_rate = 3e-3, gamma = 0.99, device: str = "mps"):
    self.device = device
    input_dim = env.observation_space.shape[0]
    print(input_dim)
    self.actor = Actor(input_dim, env.action_space.n).to(self.device)
    self.critic = Critic(input_dim).to(self.device)
    self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    self.total_rewards = np.zeros(n_episodes, dtype=np.int64)
    self.gamma = gamma
    self.n_perfect = 0

  def train_step(self, env: gym.Env, episode: int):
    if (self.n_perfect >= 10):
      return
    state = env.reset()[0]
    total_reward = 0.0
    for _ in range(10000):
      state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      action_probs = self.actor(state_tensor)
      action = np.random.choice(env.action_space.n, p=action_probs.cpu().detach().numpy()[0])
      next_state, reward, terminated, truncated, _ = env.step(action)
      total_reward += float(reward)

      next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

      state_v = self.critic(state_tensor)
      if terminated or truncated:
          next_state_v = torch.tensor(0.0, dtype=torch.float32, device=self.device)
      else:
          next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
          with torch.no_grad():
              next_state_v = self.critic(next_state_tensor)

      advantage = reward + self.gamma * next_state_v - state_v

      actor_loss = -torch.log(action_probs[0, action] + 1e-8) * advantage
      critic_loss = advantage.pow(2)

      self.a_opt.zero_grad()
      self.c_opt.zero_grad()

      total_loss = (actor_loss + critic_loss)
      total_loss.backward()

      self.a_opt.step()
      self.c_opt.step()

      state = next_state
      if (terminated or truncated):
        if (total_reward == 500):
          self.n_perfect += 1
        else:
          self.n_perfect = 0
        break
    if (episode % 10 == 0):
      print(f'Episode {episode}, Reward: {total_reward}')
    self.total_rewards[episode] = total_reward

  def choose_action(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    with torch.no_grad():
      action_probs_tensor = self.actor(state_tensor)
      action_probs = action_probs_tensor.cpu().numpy()[0]
    return np.argmax(action_probs)

  def save_w(self, path="ac_path.pth"):
    torch.save(self.actor.state_dict(), path)

  def load_w(self, path="ac_path.pth"):
    self.actor.load_state_dict(torch.load(path))
    self.actor.eval()


class Actor(Module):
  def __init__(self, input_dim:int, output_dim:int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.nn = Sequential(
      Linear(input_dim, 64),
      ReLU(),
      Linear(64, output_dim),
      Softmax(1)
    )
  def forward(self, x):
    return self.nn(x)

class Critic(Module):
  def __init__(self, input_dim:int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.nn = Sequential(
      Linear(input_dim, 64),
      ReLU(),
      Linear(64, 1)
    )
  def forward(self, x):
    return self.nn(x)