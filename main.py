from rl_games import QL, QFA, DQL, AC
import gymnasium as gym

FROZENLAKE_PATH = "frozenlake_q.npy"

def main():
  #play_frozenlake()
  play_pole_qfa()
  play_pole_ac()
  play_lunar()

def play_frozenlake():
  print("Playing frozenlake with Q-learning agent.")
  env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human")
  ql = QL(env.observation_space.n, env.action_space.n)
  ql.load_model(FROZENLAKE_PATH)
  terminated = False
  truncated = False
  reward = 0
  state = env.reset()[0]
  while not(terminated or truncated):
    state, r, terminated, truncated, _i = env.step(ql.choose_action(state))
    reward += float(r)
  print(f"{reward=}")

def play_pole_qfa():
  print("Playing Cart Pole with Q-learning with Function Approximation agent.")
  env = gym.make("CartPole-v1", render_mode="human")
  qfa = QFA(env.observation_space.shape[0], env.action_space.n)
  qfa.load_model("pole_qfa.npy")
  terminated = False
  truncated = False
  reward = 0
  state = env.reset()[0]
  while not(terminated or truncated):
    state, r, terminated, truncated, _i = env.step(qfa.choose_action(state))
    reward += float(r)
  print(f"{reward=}")

def play_pole_ac():
  print("Playing Cart Pole with Actor-Critic agent.")
  env = gym.make("CartPole-v1", render_mode="human")
  ac = AC(env.observation_space.shape[0], env.action_space.n)
  ac.load_model("pole_ac.pth")
  terminated = False
  truncated = False
  reward = 0
  state = env.reset()[0]
  while not(terminated or truncated):
    state, r, terminated, truncated, _i = env.step(ac.choose_action(state))
    reward += float(r)
  print(f"{reward=}")

def play_lunar():
  print("Playing Lunar Lander with Deep Q Learning agent.")
  env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
      enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = "human")
  dql = DQL(env.observation_space.shape[0], env.action_space.n)
  dql.load_model("lunar_dql.pth")
  terminated = False
  truncated = False
  reward = 0
  state = env.reset()[0]
  while not(terminated or truncated):
    state, r, terminated, truncated, _i = env.step(dql.choose_action(state))
    reward += float(r)
  print(f"{reward=}")

if __name__ == "__main__":
    main()
