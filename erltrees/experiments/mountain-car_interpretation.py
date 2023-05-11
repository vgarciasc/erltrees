import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config

if __name__ == "__main__":
    tree_strs = ["\n- Car Velocity <= 0.01923\n-- Car Velocity <= -0.00015\n--- Car Position <= -0.94281\n---- RIGHT\n---- LEFT\n--- Car Position <= -0.38912\n---- RIGHT\n---- LEFT\n-- RIGHT"]

    config = get_config("mountain_car")
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]
    env = config['maker']()

    history = []
    total_rewards = []
    for tree in trees:
        history.append([])
        total_rewards.append([])

        for episode in range(1000):
            done = False
            state = env.reset()
            total_rewards[-1].append(0)

            while not done:
                action = tree.act(state)
                state, reward, done, _ = env.step(action)
                env.render()
                time.sleep(0.1)
                print(f"Car Position: {state[0]}, Car Velocity: {state[1]}")

                total_rewards[-1][-1] += reward
                history[-1].append((state, action))