import gym
import numpy as np
import matplotlib.pyplot as plt
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config

if __name__ == "__main__":
    tree_strs = ["\n- Pole Angular Velocity <= 0.25301\n-- Pole Angle <= 0.03423\n--- LEFT\n--- RIGHT\n-- RIGHT",
        "\n- Pole Angle <= -0.03739\n-- LEFT\n-- Pole Angular Velocity <= -0.30179\n--- LEFT\n--- RIGHT",
        "\n- Cart Velocity <= -0.52083\n-- RIGHT\n-- Pole Angle <= 0.04229\n--- LEFT\n--- RIGHT"]

    config = get_config("mountain_car")
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]
    env = config['maker']()

    for tree in trees:
        tree.denormalize_thresholds()
        print(tree)

    # For each tree, collect a dataset (state, action) for 100 episodes
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
                total_rewards[-1][-1] += reward

                history[-1].append((state, action))

    # Plot history
    for i, tree in enumerate(trees):
        states, actions = zip(*history[i])
        states = np.array(states)
        actions = np.array(actions)

        print(f"Average reward for tree {i}: {np.mean(total_rewards[i])}")

        plt.hist(states[:, 2], density=True, bins=20, alpha=0.5, label=f"Tree {['A','B','C'][i]}")

    plt.xlabel("Pole Angle")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()