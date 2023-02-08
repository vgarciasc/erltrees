import json
import os
import argparse
import psutil
from datetime import datetime
import pdb
import gym
import time
import numpy as np
from rich import print
from erltrees.evo.evo_tree import Individual
import erltrees.rl.configs as configs
import erltrees.rl.utils as rl
from scipy.stats import ks_2samp, anderson_ksamp
from erltrees.io import printv, console

def save_history_to_file(filepath, history):
    with open(filepath, "a+") as file:
        for row in history:
            file.write(";".join([str(r) for r in row]) + "\n")

def save_trees_to_file(filepath, original_trees, trees, prefix):
    string = prefix

    rewards = [tree.reward for tree in trees]
    sizes = [tree.get_tree_size() for tree in trees]
    success_rates = [tree.success_rate for tree in trees]

    string += f"\n\n"
    string += f"Mean Best Reward: {np.mean(rewards)} +- {np.std(rewards)}\n"
    string += f"Mean Best Size: {np.mean(sizes)}\n"
    string += f"Average Evaluations to Success: -------\n"
    string += f"Success Rate: {np.mean(success_rates)}\n"
    string += "\n-----\n\n"

    for i, tree in enumerate(trees):
        original_tree = original_trees[i]
        string += f"Tree #{i} (Reward: {'{:.3f}'.format(tree.reward)} +- {'{:.3f}'.format(tree.std_reward)}, Success Rate: {tree.success_rate}, Size: {tree.get_tree_size()}); elapsed time: {'{:.2f}'.format(tree.elapsed_time)} seconds; ((original tree #{original_tree.orig_id}))\n"
        string += "----------\n"
        string += str(tree)
        string += "\n"

    with open(filepath, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def fill_tree_given_data(tree, rewards, alpha, task_solution_threshold):
    tree.fitness = rl.calc_fitness(np.mean(rewards), np.std(rewards), tree.get_tree_size(), alpha, should_penalize_std=True)
    tree.success_rate = np.mean([(1 if r > task_solution_threshold else 0) for r in rewards])
    tree.reward = np.mean(rewards)
    tree.std_reward = np.std(rewards)

def reward_pruning(tree, node, config, episodes=100, alpha=0,
    task_solution_threshold=0, should_norm_state=True, 
    should_use_kstest=True, n_jobs=4, verbose=False,
    kstest_threshold=0.1, run_id="default"):

    tree = tree.copy()
    history = []
    
    nodes = tree.get_node_list(get_inners=True, get_leaves=False)
    # nodes = filter(lambda x : x.get_tree_size() == 3, nodes)
    nodes.sort(key=lambda x : x.get_tree_size())
    node_paths = [node.get_path() for node in nodes]
    node_paths = node_paths[:-1] #shouldn't try to remove root

    rewards_curr = rl.collect_rewards_par(config, tree, episodes, should_norm_state, n_jobs=n_jobs)
    fill_tree_given_data(tree, rewards_curr, alpha, task_solution_threshold)

    for node_path in node_paths:
        printv("-----------------------", verbose)
        printv(f"-- Pruning a tree with {tree.get_tree_size()} nodes.", verbose)
        process = psutil.Process(os.getpid())
        # printv(f'-- RAM %: {process.memory_percent()}', verbose)

        tree_alt_1 = tree.copy()
        node = tree_alt_1.get_node_by_path(node_path)
        node.left.cut_parent()

        rewards_alt_1 = rl.collect_rewards_par(config, tree_alt_1, episodes, should_norm_state, n_jobs=n_jobs)
        fill_tree_given_data(tree_alt_1, rewards_alt_1, alpha, task_solution_threshold)

        printv(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its left node.", verbose)
        printv(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}. (size: {tree.get_tree_size()}, fit: {tree.fitness}, sr: {tree.success_rate})", verbose)
        printv(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_1))} +- {'{:.3f}'.format(np.std(rewards_alt_1))}. (size: {tree_alt_1.get_tree_size()}, fit: {tree_alt_1.fitness}, sr: {tree_alt_1.success_rate})", verbose)

        stats, pvalue = ks_2samp(rewards_curr, rewards_alt_1)
        printv(f"------ KL Stat: {stats}, P-value: {pvalue}", verbose)

        if (tree_alt_1.fitness > tree.fitness) or (tree_alt_1.success_rate > tree.success_rate) or (stats < kstest_threshold and should_use_kstest):
            printv(f"------ [green]Maintaining change.[/green] ([yellow]{'fitness' if tree_alt_1.fitness > tree.fitness else ('success rate' if tree_alt_1.success_rate > tree.success_rate else 'kstest')}[/yellow])", verbose)
            tree = tree_alt_1
            rewards_curr = rewards_alt_1
        else:
            printv(f"------ [red]Undoing change.[/red]", verbose)

            tree_alt_2 = tree.copy()
            node = tree_alt_2.get_node_by_path(node_path)
            node.right.cut_parent()

            rewards_alt_2 = rl.collect_rewards_par(config, tree_alt_2, episodes, should_norm_state, n_jobs=n_jobs)
            fill_tree_given_data(tree_alt_2, rewards_alt_2, alpha, task_solution_threshold)
            printv(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its right node.", verbose)
            printv(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}. (size: {tree.get_tree_size()}, fit: {tree.fitness}, sr: {tree.success_rate})", verbose)
            printv(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_2))} +- {'{:.3f}'.format(np.std(rewards_alt_2))}. (size: {tree_alt_2.get_tree_size()}, fit: {tree_alt_2.fitness}, sr: {tree_alt_2.success_rate})", verbose)
            
            stats, pvalue = ks_2samp(rewards_curr, rewards_alt_2)
            printv(f"------ KL Stat: {stats}, P-value: {pvalue}", verbose)

            if (tree_alt_2.fitness > tree.fitness) or (tree_alt_2.success_rate > tree.success_rate) or (stats < kstest_threshold and should_use_kstest):
                printv(f"------ [green]Maintaining change.[/green] ([yellow]{'fitness' if tree_alt_2.fitness > tree.fitness else ('success rate' if tree_alt_2.success_rate > tree.success_rate else 'kstest')}[/yellow])", verbose)
                tree = tree_alt_2
                rewards_curr = rewards_alt_2
            else:
                printv(f"------ [red]Undoing change.[/red]", verbose)
        
        history.append((run_id, tree.get_tree_size(), tree.reward, tree.std_reward, tree.fitness, tree.success_rate))

    return tree, history        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Pruning')
    parser.add_argument('-t','--task', help="Which task to execute?", type=str, required=True)
    parser.add_argument('-f','--input', help="Which file to use as input?", type=str, required=True)
    parser.add_argument('-a','--alpha', help='Which alpha to use?', required=True, default=1.0, type=float)
    parser.add_argument('--should_use_kstest', help='Should use KS test to detect if trees are equal?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--kstest_threshold', help='Which KS test threshold to use?', required=False, default=0.1, type=float)
    parser.add_argument('--rounds', help='How many rounds for reward pruning?', required=True, default=1, type=int)
    parser.add_argument('--simulations', help='How many simulations to run?', required=False, default=-1, type=int)
    parser.add_argument('--episodes', help='How many episodes to run?', required=True, type=int)
    parser.add_argument('--norm_state', help="Should normalize state?", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_jobs', help="How many jobs to run?", type=int, required=False, default=-1)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=True, default=None, type=int)
    args = vars(parser.parse_args())

    filepath = "data/reward_pruning_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt"
    history_filepath = "data/reward_pruning_log.txt"

    command_line = str(args)
    command_line += "\n\npython -m erltrees.experiments.reward_pruning " + " ".join([f"--{key} {val}" for (key, val) in args.items()])
    command_line += "\n\n"

    config = configs.get_config(args["task"])

    with open(args["input"]) as f:
        tree_strs = json.load(f)
    
    original_trees = []
    final_trees = []

    if args['simulations'] == -1:
        simulations = len(tree_strs)
    else:
        simulations = args["simulations"]
    
    for simulation_id in range(simulations):
        # Initialization
        run_id = datetime.now().strftime("%Y-%m-%d_%I-%M-%S")

        orig_id = simulation_id % len(tree_strs)
        tree_str = tree_strs[orig_id]
        
        original_tree = Individual.read_from_string(config, tree_str)
        original_tree.orig_id = orig_id

        original_trees.append(original_tree)

        tree = original_tree.copy()
        if args["norm_state"]:
            tree.denormalize_thresholds()

        START_TIME = time.time()
        # Running RP
        history = []
        for round in range(args["rounds"]):
            console.rule(f"Round {round + 1} / {args['rounds']}, Simulation #{simulation_id + 1} / {args['simulations']}")
            
            tree, hist = reward_pruning(tree, tree, config, episodes=args["episodes"], alpha=args["alpha"], 
                should_norm_state=False, should_use_kstest=args["should_use_kstest"], n_jobs=args["n_jobs"],
                task_solution_threshold=args["task_solution_threshold"], kstest_threshold=args["kstest_threshold"],
                verbose=True, run_id=run_id)
            
            history += hist
        END_TIME = time.time()
        tree.elapsed_time = END_TIME - START_TIME

        # Evaluating final tree
        rl.collect_metrics(config, [tree], args["alpha"], args["episodes"], should_norm_state=False, penalize_std=True, 
            should_fill_attributes=True, task_solution_threshold=args["task_solution_threshold"], n_jobs=args["n_jobs"])
        final_trees.append(tree)

        # Housekeeping
        save_trees_to_file(filepath, original_trees, final_trees, command_line)
        save_history_to_file(history_filepath, history)
