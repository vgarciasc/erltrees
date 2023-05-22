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
    parser.add_argument('-o','--output', help='Output file', required=False, default=None, type=str)
    parser.add_argument('-e','--episodes', help='Episodes to run', required=False, default=100, type=int)
    args = vars(parser.parse_args())

    config = get_config(args['task'])
    filename = args['filename']
    
    with open(filename, "r") as f:
        tree_strs = json.load(f)
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]

    print(f"Running... {filename}")

    history = []
    for i, tree in enumerate(trees):
        fill_metrics(config, [tree], alpha=0.0, episodes=args['episodes'], should_norm_state=False, penalize_std=False, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)
        print(f"\t[{str.rjust(str(i), 2, ' ')}/{len(trees)}]: Reward: {tree.reward:.4f} ± {tree.std_reward:.4f}, SR: {tree.success_rate:.4f}, Size: {tree.get_tree_size()}")
        history.append((tree, tree.reward, tree.get_tree_size(), tree.success_rate))

    print(f"[red]AVERAGE[/red]: reward: {np.mean([t.reward for t in trees])} ± {np.mean([t.std_reward for t in trees])}, SR: {np.mean([t.success_rate for t in trees])}, size: {np.mean([t.get_tree_size() for t in trees])}")

    # Save to file
    if args['output'] is not None:
        string = f"Reevaluation of {filename}"
        string += "\n\npython -m erltrees.dagger_file_evaluator " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
        trees, rewards, sizes, successes = zip(*history)
        trees = np.array(trees)

        string += f"Mean Best Reward: {np.mean(rewards)} ± {np.std(rewards)}\n"
        string += f"Mean Best Size: {np.mean(sizes)}\n"
        string += f"Average Evaluations to Success: -------\n"
        string += f"Success Rate: {np.mean(successes)}\n"
        string += "\n-----\n\n"

        for i, tree in enumerate(trees):
            string += f"Tree #{i} (Reward: {tree.reward:.5f} ± {tree.std_reward:.5f}, Size: {tree.get_tree_size()}, Success Rate: {tree.success_rate:.5f})\n"
            string += "----------\n"
            string += str(tree)
            string += "\n"

        with open(args['output'], "w", encoding="utf-8") as text_file:
            text_file.write(string)