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

def get_trees_from_logfile(filepath):
    tree_strings = []
    with open(filepath) as f:
        curr_line_idx = 0
        lines = f.readlines()

        while curr_line_idx < len(lines):
            if "Tree #" in lines[curr_line_idx] or "CRO-DT-RL (" in lines[curr_line_idx]:
                curr_line_idx += 2
                start_line = curr_line_idx
                while curr_line_idx < len(lines)-1 and lines[curr_line_idx] != "\n":
                    curr_line_idx += 1
                end_line = curr_line_idx
                tree_string = "\n" + "".join(lines[start_line:end_line]).rstrip()
                tree_strings.append(tree_string)
            else:
                curr_line_idx += 1
    return tree_strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reevaluator')
    parser.add_argument('-t','--task', help="Which task to run?", required=True)
    parser.add_argument('-f','--file', help="Input file", required=True, type=str)
    parser.add_argument('-e','--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--episodes_to_evaluate_best', help='Number of episodes to run when evaluating best model at the end', required=False, default=100, type=int)
    parser.add_argument('--norm_state', help="Should normalize state?", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--alpha', help="Which alpha to use?", required=False, default=1.0, type=float)
    parser.add_argument('--select_tree', help="Should select a single tree?", required=False, default=None, type=int)
    parser.add_argument('--should_prune_by_visits', help='Should prune trees by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_print_tree', help='Should print tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--denormalize_thresholds', help='Should denormalize thresholds?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_jobs', help="How many jobs to run?", type=int, required=False, default=-1)
    args = vars(parser.parse_args())

    config = configs.get_config(args['task'])
    norm_state = args['norm_state']
    
    history = []
    command_line = "\n\npython -m erltrees.reevaluator " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    reeval_filename = args['file'][:-4] + "_reevaluated.txt"
    print(f"Saving reevaluation to {reeval_filename}.")

    tree_strings = get_trees_from_logfile(args['file'])

    tree_sizes = []
    avg_rewards_before_pruning = []
    std_rewards_before_pruning = []
    success_rates_before_pruning = []
    avg_rewards_after_pruning = []
    std_rewards_after_pruning = []
    success_rates_after_pruning = []

    best_tree = None

    if args["select_tree"] is not None:
        print(f"Selecting tree #{args['select_tree']}")
        tree_strings = [tree_strings[args["select_tree"]]]

    for i, string in enumerate(tree_strings):
        io.console.rule(f"Evaluating Tree {i} / {len(tree_strings) - 1}")
        tree = evo_tree.Individual.read_from_string(config, string=string)

        print("[yellow]> Evaluating fitness:[/yellow]")
        print(f"Tree size: {tree.get_tree_size()} nodes")

        if args['should_print_tree']:
            print(tree)

        rl.collect_metrics(config, trees=[tree], alpha=args["alpha"],
            task_solution_threshold=config["task_solution_threshold"],
            episodes=args['episodes'], should_norm_state=norm_state,
            should_fill_attributes=True, penalize_std=True,
            n_jobs=-1)
        print(f"Mean reward, std reward: {tree.reward} ± {tree.std_reward}, fitness: {tree.fitness}, SR: {tree.success_rate}")

        avg_rewards_before_pruning.append(tree.reward)
        std_rewards_before_pruning.append(tree.std_reward)
        success_rates_before_pruning.append(tree.success_rate)

        if args["should_prune_by_visits"]:
            print("[yellow]> Pruning by visits...[/yellow]")
            tree = tree.prune_by_visits(5)
            print(f"Tree size: {tree.get_tree_size()} nodes")

            rl.collect_metrics(config, trees=[tree], alpha=args["alpha"],
                task_solution_threshold=config["task_solution_threshold"],
                episodes=args['episodes'], should_norm_state=norm_state,
                should_fill_attributes=True, penalize_std=True,
                n_jobs=args["n_jobs"])
            print(f"Mean reward, std reward: {tree.reward} ± {tree.std_reward}, fitness: {tree.fitness}, SR: {tree.success_rate}")

        tree_sizes.append(tree.get_tree_size())

        avg_rewards_after_pruning.append(tree.reward)
        std_rewards_after_pruning.append(tree.std_reward)
        success_rates_after_pruning.append(tree.success_rate)

        #Denormalize thresholds
        if args["denormalize_thresholds"]:
            tree.denormalize_thresholds()

        tree.elapsed_time = -1
        history.append((tree, tree.reward, tree.get_tree_size(), tree.success_rate))

        io.save_history_to_file(config, history, reeval_filename, None, command_line)

        if best_tree is None or tree.fitness > best_tree.fitness:
            best_tree = tree
            best_tree.best_id = i

    if args["select_tree"] is not None:
        print(tree)

    io.console.rule("[red]SUMMARY[/red]")
    print(f"[green]File: \"{args['file']}\"[/green]")
    if args["should_prune_by_visits"]:
        print("Before pruning:")
        print(f"  Average reward before pruning: {'{:.3f}'.format(np.mean(avg_rewards_before_pruning))} ± {'{:.3f}'.format(np.mean(std_rewards_before_pruning))}")
        print(f"  Average success rate before pruning: {'{:.3f}'.format(np.mean(success_rates_before_pruning))} ± {'{:.3f}'.format(np.std(success_rates_before_pruning))}")
        print()
        print("After pruning:")
        print(f"  Average reward: {'{:.3f}'.format(np.mean(avg_rewards_after_pruning))} ± {'{:.3f}'.format(np.mean(std_rewards_after_pruning))}")
        print(f"  Average success rate: {'{:.3f}'.format(np.mean(success_rates_after_pruning))} ± {'{:.3f}'.format(np.std(success_rates_after_pruning))}")
        print(f"  Average tree size: {'{:.3f}'.format(np.mean(tree_sizes))} ± {'{:.3f}'.format(np.std(tree_sizes))}")
    else:
        print(f"  Average reward: {'{:.3f}'.format(np.mean(avg_rewards_before_pruning))} ± {'{:.3f}'.format(np.mean(std_rewards_before_pruning))}")
        print(f"  Average success rate: {'{:.3f}'.format(np.mean(success_rates_before_pruning))} ± {'{:.3f}'.format(np.std(success_rates_before_pruning))}")
        print(f"  Average tree size: {'{:.3f}'.format(np.mean(tree_sizes))} ± {'{:.3f}'.format(np.std(tree_sizes))}")
    
    rl.collect_metrics(config, trees=[best_tree], alpha=args["alpha"],
        task_solution_threshold=config["task_solution_threshold"],
        episodes=args["episodes_to_evaluate_best"], should_norm_state=False, 
        should_fill_attributes=True, penalize_std=True,
        n_jobs=args["n_jobs"])
    
    print()
    print(f"Best tree (pocket): tree #{best_tree.best_id}")
    print(f"  Tree size: {best_tree.get_tree_size()} nodes")
    print(f"  Mean reward, std reward: {best_tree.reward} ± {best_tree.std_reward}")
    print(f"  Fitness: {best_tree.fitness}")
    print(f"  Success Rate: {best_tree.success_rate}")
    