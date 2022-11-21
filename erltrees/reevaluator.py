import argparse
from copy import deepcopy
import math
import time
import numpy as np
import pdb
import matplotlib.pyplot as plt
from rich import print

import erltrees.evo.evo_tree as evo_tree
import erltrees.rl.utils as rl
import erltrees.rl.configs as configs
import erltrees.io as io

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary Programming')
    parser.add_argument('--task', help="Which task to run?", required=True)
    parser.add_argument('--file', help="Input file", required=True, type=str)
    parser.add_argument('--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--norm_state', help="Should normalize state?", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--select_tree', help="Should select a single tree?", required=False, default=None, type=int)
    parser.add_argument('--task_solution_threshold', help="At which reward is the episode considered solved?", required=True, type=int)
    parser.add_argument('--should_print_tree', help='Should print tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_jobs', help="How many jobs to run?", type=int, required=False, default=-1)
    args = vars(parser.parse_args())

    config = configs.get_config(args['task'])
    task_solution_threshold = args['task_solution_threshold']
    norm_state = args['norm_state']
    alpha = 1.0

    tree_strings = io.get_trees_from_logfile(args['file'])

    tree_sizes = []
    avg_rewards_before_pruning = []
    std_rewards_before_pruning = []
    success_rates_before_pruning = []
    avg_rewards_after_pruning = []
    std_rewards_after_pruning = []
    success_rates_after_pruning = []

    if args["select_tree"] is not None:
        print(f"Selecting tree #{args['select_tree']}")
        tree_strings = [tree_strings[args["select_tree"]]]

    for i, string in enumerate(tree_strings):
        io.console.rule(f"Evaluating Tree {i} / {len(tree_strings) - 1}")
        tree = evo_tree.Individual.read_from_string(config, string=string)

        if norm_state:
            tree.denormalize_thresholds()

        print("[yellow]> Evaluating fitness:[/yellow]")
        print(f"Tree size: {tree.get_tree_size()} nodes")

        if args['should_print_tree']:
            print(tree)

        rl.collect_metrics(config, trees=[tree], alpha=alpha,
            task_solution_threshold=args["task_solution_threshold"],
            episodes=100, should_norm_state=False, 
            should_fill_attributes=True, penalize_std=False,
            n_jobs=-1)
        print(f"Mean reward, std reward: {tree.reward} ± {tree.std_reward}, SR: {tree.success_rate}")

        avg_rewards_before_pruning.append(tree.reward)
        std_rewards_before_pruning.append(tree.std_reward)
        success_rates_before_pruning.append(tree.success_rate)

        print("[yellow]> Pruning by visits...[/yellow]")
        tree = tree.prune_by_visits(5)
        print(f"Tree size: {tree.get_tree_size()} nodes")

        rl.collect_metrics(config, trees=[tree], alpha=alpha,
            task_solution_threshold=args["task_solution_threshold"],
            episodes=args['episodes'], should_norm_state=False, 
            should_fill_attributes=True, penalize_std=False,
            n_jobs=args["n_jobs"])
        print(f"Mean reward, std reward: {tree.reward} ± {tree.std_reward}, SR: {tree.success_rate}")

        tree_sizes.append(tree.get_tree_size())

        avg_rewards_after_pruning.append(tree.reward)
        std_rewards_after_pruning.append(tree.std_reward)
        success_rates_after_pruning.append(tree.success_rate)

    if args["select_tree"] is not None:
        print(tree)

    io.console.rule("[red]SUMMARY[/red]")
    print("Before pruning:")
    print(f"  Average reward before pruning: {'{:.3f}'.format(np.mean(avg_rewards_before_pruning))} ± {'{:.3f}'.format(np.mean(std_rewards_before_pruning))}")
    print(f"  Average success rate before pruning: {'{:.3f}'.format(np.mean(success_rates_before_pruning))} ± {'{:.3f}'.format(np.std(success_rates_before_pruning))}")
    print()
    print("After pruning:")
    print(f"  Average reward: {'{:.3f}'.format(np.mean(avg_rewards_after_pruning))} ± {'{:.3f}'.format(np.mean(std_rewards_after_pruning))}")
    print(f"  Average success rate: {'{:.3f}'.format(np.mean(success_rates_after_pruning))} ± {'{:.3f}'.format(np.std(success_rates_after_pruning))}")
    print(f"  Average tree size: {'{:.3f}'.format(np.mean(tree_sizes))} ± {'{:.3f}'.format(np.std(tree_sizes))}")