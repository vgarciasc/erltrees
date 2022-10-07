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
    parser.add_argument('--reward_to_solve', help="At which reward is the episode considered solved?", required=True, type=int)
    parser.add_argument('--should_print_tree', help='Should print tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = configs.get_config(args['task'])
    reward_to_solve = args['reward_to_solve']
    norm_state = args['norm_state']

    tree_strings = io.get_trees_from_logfile(args['file'])

    successes_rates = []
    tree_sizes = []
    avg_rewards_before_pruning = []
    std_rewards_before_pruning = []
    avg_rewards_after_pruning = []
    std_rewards_after_pruning = []

    for i, string in enumerate(tree_strings):
        io.console.rule(f"Evaluating Tree {i} / {len(tree_strings) - 1}")
        tree = evo_tree.Individual.read_from_string(config, string=string)

        if norm_state:
            tree.denormalize_thresholds()

        print("[yellow]> Evaluating fitness:[/yellow]")
        print(f"Tree size: {tree.get_tree_size()} nodes")

        if args['should_print_tree']:
            print(tree)

        rewards = [rl.collect_metrics(config, trees=[tree], episodes=1, should_norm_state=False, penalize_std=False)[0][0] for _ in range(args['episodes'])]
        success_rate = np.mean([1 if r > reward_to_solve else 0 for r in rewards])
        print(f"Mean reward, std reward: {np.mean(rewards)} +- {np.std(rewards)}, SR: {success_rate}")
        avg_rewards_before_pruning.append(np.mean(rewards))
        std_rewards_before_pruning.append(np.std(rewards))

        print("[yellow]> Pruning by visits...[/yellow]")
        tree = tree.prune_by_visits(5)
        print(f"Tree size: {tree.get_tree_size()} nodes")

        rewards = [rl.collect_metrics(config, trees=[tree], episodes=1, should_norm_state=False, penalize_std=False)[0][0] for _ in range(args['episodes'])]
        success_rate = np.mean([1 if r > reward_to_solve else 0 for r in rewards])
        print(f"Mean reward, std reward: {np.mean(rewards)} +- {np.std(rewards)}, SR: {success_rate}")

        tree_sizes.append(tree.get_tree_size())
        successes_rates.append(success_rate)
        avg_rewards_after_pruning.append(np.mean(rewards))
        std_rewards_after_pruning.append(np.std(rewards))
    
    io.console.rule("[red]SUMMARY[/red]")
    print(f"Average reward before pruning: {'{:.3f}'.format(np.mean(avg_rewards_before_pruning))} +- {'{:.3f}'.format(np.mean(std_rewards_before_pruning))}")
    print(f"Average reward after pruning: {'{:.3f}'.format(np.mean(avg_rewards_after_pruning))} +- {'{:.3f}'.format(np.mean(std_rewards_after_pruning))}")
    print(f"Average success rate: {'{:.3f}'.format(np.mean(successes_rates))}")
    print(f"Average tree size: {'{:.3f}'.format(np.mean(tree_sizes))}")