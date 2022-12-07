import gym
import json
import pdb
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree
from erltrees.il.parser import handle_args

from erltrees.io import printv
from erltrees.il.behavioral_cloning import run_behavior_cloning
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl

def run_grid_behavior_cloning(config, X, y, start, end, steps, 
    episodes_to_grade=100, task_solution_threshold=None, n_jobs=-1, 
    verbose=False):

    history = []

    for i, pruning_alpha in enumerate(np.linspace(start, end, steps)):
        # Run behavior cloning for this value of pruning
        dt = run_behavior_cloning(
            config, X, y, "DistilledTree",
            pruning_alpha=pruning_alpha)

        # Evaluating tree
        rl.collect_metrics(config, [dt], episodes=episodes_to_grade, n_jobs=n_jobs,
            should_fill_attributes=True, task_solution_threshold=task_solution_threshold)
        
        # Keeping history of trees
        size = dt.get_size()
        depth = dt.model.get_depth()
        success_rate = dt.success_rate
        history.append((pruning_alpha, dt.reward, dt.std_reward, size, depth, success_rate))

        # Logging info if necessary
        printv(f"#({i} / {steps}) PRUNING = {pruning_alpha}: \t"
            + f"REWARD = {'{:.3f}'.format(dt.reward)} ± {'{:.3f}'.format(dt.std_reward)}"
            + f"\tNODES: {size}, DEPTH: {depth}, SR: {success_rate}.",
            verbose)

        # # Saving tree
        # if should_save_trees:
        #     with open(f"data/{config['name']}_bc_pruning_{pruning_alpha}", "w") as f:
        #         f.write(dt.get_as_viztree())
    
    # pruning_params, avg_rewards, deviations, sizes, depths = zip(*history)
    return zip(*history)

def plot_behavior_cloning(filepath, history):
    pruning_params, avg_rewards, deviations, sizes, depths, success_rates = history
    
    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(pruning_params, avg_rewards - deviations, avg_rewards + deviations, color="red", alpha=0.2)
    ax1.plot(pruning_params, avg_rewards, color="red")
    ax1.set_xlabel("Pruning $\\alpha$")
    ax1.set_ylabel("Average reward")
    ax2.plot(pruning_params, sizes, color="blue")
    ax2.set_ylabel("Number of sizes")
    ax2.set_xlabel("Pruning $\\alpha$")
    plt.suptitle(f"Behavior cloning for {config['name']}")
    plt.savefig(filepath)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning Grid')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-s','--start', help='Starting point for pruning alpha', required=True, type=float)
    parser.add_argument('-e','--end', help='Ending point for pruning alpha', required=True, type=float)
    parser.add_argument('-i','--steps', help='Number of overall steps', required=True, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--expert_exploration_rate', help='What is the expert exploration rate?', required=False, default=0, type=float)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=None, type=int)
    parser.add_argument('--episodes_to_grade_model', help='How many episodes to grade model?', required=False, default=100, type=int)
    parser.add_argument('--n_jobs', help='How many jobs to parallelize?', required=False, default=-1, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = get_config(args['task'])
    expert, X, y = handle_args(args, config)

    # Grid-running behavior cloning
    history = run_grid_behavior_cloning(
        config, X, y,
        start=args['start'],
        end=args['end'],
        steps=args['steps'],
        episodes_to_grade=args['episodes_to_grade_model'],
        task_solution_threshold=args['task_solution_threshold'],
        n_jobs=args["n_jobs"],
        verbose=args['verbose'])
    history = list(history)
    
    run_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    output_file = "data/imitation_learning/behavioral_cloning_grid_" + run_id + ".txt"
    with open(output_file, "w") as f:
        pruning_params, avg_rewards, deviations, sizes, depths, success_rates = history
        string = {
            "args": str(args),
            "command_line": "python -m erltrees.il.behavioral_cloning_grid " + " ".join([f"--{key} {val}" for (key, val) in args.items()]),
            "pruning_alpha": list(pruning_params),
            "avg_rewards": list(avg_rewards),
            "std_rewards": list(deviations),
            "sizes": list([float(l) for l in sizes]),
            "depths": list([float(d) for d in depths]),
            "success_rates": list([float(s) for s in success_rates])}
        json.dump(string, f, indent=1)

    # Plotting behavior cloning
    plot_behavior_cloning("data/imitation_learning/" + run_id + ".png", history)
    