import json
import gym
import pickle
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rich import print

from erltrees.il.behavioral_cloning import get_model_to_train
from erltrees.il.parser import handle_args
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl
import erltrees.il.utils as il

def run_dagger(config, X, y, model_name, pruning_alpha, 
    expert, iterations, episodes):

    best_reward = -9999
    best_model = None

    dt = get_model_to_train(config, model_name)
    dt.fit(X, y, pruning=pruning_alpha)

    history = []

    for i in range(iterations):
        # Collect trajectories from student and correct them with expert
        X2, _ = il.get_dataset_from_model(config, dt, episodes)
        y2 = il.label_dataset_with_model(expert, X2)

        # Aggregate datasets
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))

        # Sample from dataset aggregation
        # D = list(zip(X, y))
        # D = random.sample(D, args['dataset_size'])
        # X, y = zip(*D)

        # Train new student
        dt = get_model_to_train(config, model_name)
        dt.fit(X, y, pruning=pruning_alpha)

        # Housekeeping
        print(f"Step #{i}.")

        rl.collect_metrics(config, [dt], episodes=iterations, should_fill_attributes=True)
        print(f"Average reward is {dt.reward} ± {dt.std_reward}.")

        print(f"- Dataset length: {len(X)}")
        print(f"- Obtained tree with {dt.get_size()} nodes.")
        print()

        history.append((i, dt.reward, dt.std_reward, dt.get_size(), dt))

        if dt.reward > best_reward:
            best_reward = dt.reward
            best_model = dt
    
    return best_model, best_reward, zip(*history)

def plot_dagger(config, avg_rewards, deviations, nodes, pruning_alpha, episodes, show=False):
    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations,
        color="red", alpha=0.2)
    ax1.plot(iterations, avg_rewards, color="red")
    ax1.set_ylabel("Average reward")
    ax1.set_xlabel("Iterations")
    ax2.plot(iterations, nodes, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Iterations")
    plt.suptitle(f"DAgger for {config['name']} w/ pruning $\\alpha = {pruning_alpha}$" +
        f", {episodes} per iteration")
    
    if show:
        plt.show()
    else:
        plt.savefig(f"figures/dagger_{config['name']}_{pruning_alpha}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-c','--class', help='Model to use', required=True)
    parser.add_argument('-e','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-j','--episodes', help='Number of episodes to collect every iteration', required=True, type=int)
    parser.add_argument('--episodes_to_evaluate', help='Number of episodes to run when evaluating best model', required=False, default=100, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--expert_exploration_rate', help='The epsilon to use during dataset collection', required=False, default=0.0, type=float)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save_models', help='Should save trees?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_models_pathname', help='Where to save trees?', required=False, default="imitation_learning/models/tmp.txt", type=str)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=0, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = get_config(args['task'])
    expert, X, y = handle_args(args, config)
    
    print(f"Running {args['class']} DAgger for {config['name']} with pruning = {args['pruning']}.")
    # Running DAgger
    dt, reward, history = run_dagger(
        config, X, y,
        pruning_alpha=args['pruning'],
        model_name=args['class'],
        expert=expert, 
        iterations=args['iterations'],
        episodes=args['episodes'])
    iterations, avg_rewards, deviations, model_sizes, models = history

    if args['should_save_models']:
        model_strs = [model.get_as_viztree() for model in models]
        
        with open(args['save_models_pathname'], "w") as f:
            json.dump(model_strs, f)

    # Plotting results
    if args['should_plot']:
        plot_dagger(
            config=config,
            avg_rewards=avg_rewards,
            deviations=deviations,
            nodes=model_sizes,
            pruning_alpha=args['pruning'],
            episodes=args['episodes'],
            show=args['should_plot'])

    # Printing the best model
    rewards = [rl.collect_metrics(config, trees=[dt], episodes=1)[0][0] for _ in range(args['episodes_to_evaluate'])]
    success_rate = np.mean([1 if r > args["task_solution_threshold"] else 0 for r in rewards])

    print()
    print(f"- Average reward for the best policy is {np.mean(rewards)} ± {np.std(rewards)}.")
    print(f"- Success rate is {success_rate}.")

    print(f"- Obtained tree with {dt.get_size()} nodes.")

    # Visualizing the best model
    if args['should_visualize']:
        print(f"Visualizing final tree:")
        rl.collect_metrics(config, [dt], episodes=10, 
            should_norm_state=False, render=True, verbose=True)

    # Saving best model
    dt.save_model(f"data/dagger_best_tree_{config['name']}")
    date = datetime.now().strftime("tree_%Y-%m-%d_%H-%M")
    filename = f"data/{config['name']}_{date}_dagger_{args['pruning']}"

    print(dt.get_as_viztree())
        