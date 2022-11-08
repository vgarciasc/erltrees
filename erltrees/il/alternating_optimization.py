import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich import print

import ann
import imitlearn.env_configs
import imitlearn.parser
from imitlearn.il import *
from qtree import save_tree_from_print
from imitlearn.utils import load_dataset, printv, save_dataset
from imitlearn.distilled_tree import DistilledTree
from imitlearn.keras_dnn import KerasDNN

def run_altopt(config, X, y, expert, pruning_alpha, 
    iterations, episodes, verbose=False):

    best_reward = -9999
    best_model = None

    dt = DistilledTree(config)
    dt.fit(X, y, pruning=args['pruning'])

    history = []
    for i in range(iterations):
        X, _ = get_dataset_from_model(config, dt, episodes)
        y = label_dataset_with_model(config, expert, X)

        dt = DistilledTree(config)
        dt.fit(X, y, pruning=pruning_alpha)

        printv(f"Step #{i}.", verbose)
        avg_reward, rewards = get_average_reward(config, dt)
        deviation = np.std(rewards)
        leaves = dt.model.get_n_leaves()
        depth = dt.model.get_depth()

        printv(f"- Obtained tree with {leaves} leaves and depth {depth}.", verbose)
        printv(f"- Average reward for the student: {avg_reward} ± {deviation}.", verbose)

        history.append((i, avg_reward, deviation, leaves, depth))

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = dt
    
    return best_model, best_reward, zip(*history)

def plot_altopt(config, pruning, history):
    iterations, avg_rewards, deviations, leaves, depths = history

    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations, color="red", alpha=0.2)
    ax1.plot(iterations, avg_rewards, color="red")
    ax1.set_xlabel("Pruning $\\alpha$")
    ax1.set_ylabel("Average reward")
    ax2.plot(iterations, leaves, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Pruning $\\alpha$")
    plt.suptitle(f"Alternating Optimization for {config['name']} w/ pruning $\\alpha = {args['pruning']}$")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-e','--episodes', help='Number of episodes to collect every iteration', required=True, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitlearn.env_configs.get_config(args['task'])
    expert, X, y = imitlearn.parser.handle_args(args, config)
    
    # Running alternating optimization
    dt, reward, history = run_altopt(
        config, X, y, expert,
        pruning_alpha=args['pruning'],
        episodes=args['episodes'],
        iterations=args['iterations'],
        verbose=args['verbose'])

    plot_altopt(config, args['pruning'], history)

    # Printing the best model
    avg_reward, rewards = get_average_reward(config, dt)
    deviation = np.std(rewards)
    leaves = dt.model.get_n_leaves()
    depth = dt.model.get_depth()
    printv(f"- Obtained tree with {leaves} leaves and depth {depth}.")
    printv(f"- Average reward for the best policy: {avg_reward} ± {deviation}.")
    
    # Visualizing tree
    dt.save_fig()
    dt.save_model(f"data/AltOpt_best_tree_{config['name']}")
    if args['should_visualize']:
        printv(f"Visualizing final tree:")
        visualize_model(config, dt, 25)

    # Saving tree
    qtree = dt.get_as_qtree()
    save_tree_from_print(
        qtree,
        config['actions'],
        f"_AltOpt_{config['name']}")