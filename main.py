from games import pole, frozenlake, lunar_lander_disc

def main():
  is_slippery = False
  #frozenlake.train(n_episodes=15000, is_slippery=is_slippery)
  #frozenlake.play(n=1, is_slippery=is_slippery, render="human")
  #pole.train(1000, strategy = pole.PoleStrategy.AC)
  #pole.play(n=1, strategy = pole.PoleStrategy.AC, render="human")
  ###lunar_lander_disc.train(1000)
  lunar_lander_disc.play(1, "human")


if __name__ == "__main__":
    main()
