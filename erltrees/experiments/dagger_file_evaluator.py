from rich import print
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
    parser = argparse.ArgumentParser(description='Dagger File Evaluator')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Model to use', required=True)
    parser.add_argument('-e','--episodes', help='Episodes to run', required=False, default=100, type=int)
    args = vars(parser.parse_args())

    config = get_config(args['task'])
    filename = args['filename']
    
    with open(filename, "r") as f:
        tree_strs = json.load(f)
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]

    print(f"Running... {filename}")
    
    for i, tree in enumerate(trees):
        fill_metrics(config, [tree], alpha=0.0, episodes=args['episodes'], should_norm_state=False, penalize_std=False, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)
        print(f"\t[{i}/{len(trees)}]: Reward: {tree.reward} +- {tree.std_reward}, SR: {tree.success_rate}, Size: {tree.get_tree_size()}")

    print(f"[red]AVERAGE[/red]: reward: {np.mean([t.reward for t in trees])} +- {np.mean([t.std_reward for t in trees])}, SR: {np.mean([t.success_rate for t in trees])}, size: {np.mean([t.get_tree_size() for t in trees])}")
    pdb.set_trace()