import gym
from erltrees.evo.evo_tree import Individual

import pdb
import erltrees.trees as trees
import erltrees.rl.configs as configs
import erltrees.rl.utils as rl
import numpy as np

if __name__ == "__main__":
    tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- Car Position <= 2.000\n---- RIGHT\n---- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    config = configs.get_config("mountain_car")
    tree = Individual.read_from_string(config, tree_str)

    rl.fill_metrics(config, [tree], 0, 1000, n_jobs=8, should_norm_state=True)
    print(f"Tree has reward {tree.reward} +- {tree.std_reward}.")

    print(tree)

    m = tree.get_leaf_mask()
    print(m)

    W = tree.get_weight_matrix()
    print(W)

    labels = tree.get_label_vector()
    print(labels)

    env = gym.make(config["name"])
    state = env.reset()