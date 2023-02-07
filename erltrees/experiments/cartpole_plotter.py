import pdb
import json
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import argparse
from matplotlib.colors import ListedColormap
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.rl.utils import fill_metrics

if __name__ == "__main__":
    config = get_config("cartpole")
    with open("models/dagger_cp_population.txt", "r") as f:
        tree_strs = json.load(f)
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]

    fill_metrics(config, trees, alpha=0.0, episodes=10, should_norm_state=False, penalize_std=False, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)
    
    print(f"average reward: {np.mean([t.reward for t in trees])} +- {np.mean([t.std_reward for t in trees])}, SR: {np.mean([t.success_rate for t in trees])}, size: {np.mean([t.get_tree_size() for t in trees])}")
    pdb.set_trace()