import gym
import pickle
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree
from erltrees.il.parser import handle_args

from erltrees.io import printv
from erltrees.il.dagger import run_dagger
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl

def run_grid_dagger(config, X, y, expert, start, end, steps,
    dagger_iterations=50, dagger_episodes=100,
    episodes_to_grade=100, verbose=False):
    history = []

    for i, pruning_alpha in enumerate(np.linspace(start, end, steps)):
        # Run behavior cloning for this value of pruning
        dt, _, _ = run_dagger(config, X, y, "DistilledTree", expert, alpha=pruning_alpha, 
            iterations=dagger_iterations, episodes=dagger_episodes, should_penalize_std=True,
            task_solution_threshold=None, should_attenuate_alpha=False)

        # Evaluating tree
        rl.collect_metrics(config, [dt], episodes=episodes_to_grade, should_fill_attributes=True)
        
        # Keeping history of trees
        size = dt.get_size()
        depth = dt.model.get_depth()
        history.append((pruning_alpha, dt.reward, dt.std_reward, size, depth))

        # Logging info if necessary
        printv(f"#({i} / {steps}) PRUNING = {pruning_alpha}: \t"
            + f"REWARD = {'{:.3f}'.format(dt.reward)} ± {'{:.3f}'.format(dt.std_reward)}"
            + f"\tNODES: {size}, DEPTH: {depth}.",
            verbose)

        # # Saving tree
        # if should_save_trees:
        #     with open(f"data/{config['name']}_bc_pruning_{pruning_alpha}", "w") as f:
        #         f.write(dt.get_as_viztree())
    
    # pruning_params, avg_rewards, deviations, leaves, depths = zip(*history)
    return zip(*history)

def plot_behavior_cloning(history):
    pruning_params, avg_rewards, deviations, leaves, depths = history
    
    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(pruning_params, avg_rewards - deviations, avg_rewards + deviations, color="red", alpha=0.2)
    ax1.plot(pruning_params, avg_rewards, color="red")
    ax1.set_xlabel("Pruning $\\alpha$")
    ax1.set_ylabel("Average reward")
    ax2.plot(pruning_params, leaves, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Pruning $\\alpha$")
    plt.suptitle(f"DAgger for {config['name']}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning Grid')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-s','--start', help='Starting point for pruning alpha', required=True, type=float)
    parser.add_argument('-e','--end', help='Ending point for pruning alpha', required=True, type=float)
    parser.add_argument('-i','--steps', help='Number of overall steps', required=True, type=int)
    parser.add_argument('--dagger_iterations', help='Number of iterations to run in each dagger simulation', required=True, type=int)
    parser.add_argument('--dagger_episodes', help='Number of episodes to collect every iteration in each dagger simulation', required=True, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--expert_exploration_rate', help='What is the expert exploration rate?', required=False, default=0, type=float)
    parser.add_argument('--episodes_to_grade_model', help='How many episodes to grade model?', required=False, default=100, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = get_config(args['task'])
    expert, X, y = handle_args(args, config)

    # Grid-running behavior cloning
    history = run_grid_dagger(
        config, X, y,
        expert=expert,
        dagger_iterations=args["dagger_iterations"],
        dagger_episodes=args["dagger_episodes"],
        start=args['start'],
        end=args['end'],
        steps=args['steps'],
        episodes_to_grade=args['episodes_to_grade_model'],
        verbose=args['verbose'])

    # Plotting behavior cloning
    plot_behavior_cloning(history)
    