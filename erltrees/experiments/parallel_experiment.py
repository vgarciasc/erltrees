import json
from math import sqrt
import pdb
from joblib import Parallel, delayed
from time import time
import numpy as np
from rich import print
import argparse

from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl
from rich.console import Console

console = Console()

def run_parallel_experiment_1(alpha=0.5, episodes=30, num_trees=100):
    # Initialize population

    # tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- Car Position <= 2.000\n---- RIGHT\n---- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    # tree_str = "\n- Car Velocity <= -0.02401\n-- LEFT\n-- Car Velocity <= 0.20463\n--- Car Position <= -0.17120\n---- RIGHT\n---- LEFT\n--- RIGHT"
    # tree_str = "\n- Car Velocity <= -0.04286\n-- LEFT\n-- Car Velocity <= 0.07000\n--- Car Velocity <= 0.46567\n---- Car Position <= -0.30667\n----- RIGHT\n----- Car Position <= -0.21000\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"
    # config = get_config("mountain_car")

    tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Angle <= -0.01814\n--- Y Velocity <= -0.04140\n---- Angular Velocity <= -0.03520\n----- LEFT ENGINE\n----- X Velocity <= -0.01060\n------ MAIN ENGINE\n------ Angular Velocity <= 0.02240\n------- LEFT ENGINE\n------- X Velocity <= 0.05960\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n---- NOP\n--- Y Velocity <= 0.25919\n---- X Velocity <= -0.01300\n----- Y Velocity <= -0.05960\n------ Leg 1 is Touching <= -0.16376\n------- MAIN ENGINE\n------- NOP\n------ X Velocity <= -0.02680\n------- RIGHT ENGINE\n------- RIGHT ENGINE\n----- Y Position <= 0.16467\n------ Y Position <= 0.72484\n------- Y Position <= 1.03512\n-------- X Velocity <= 0.79464\n--------- Y Velocity <= -0.02780\n---------- MAIN ENGINE\n---------- NOP\n--------- MAIN ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n------ Y Velocity <= -0.04700\n------- Angle <= 0.07257\n-------- X Position <= -0.01733\n--------- Angular Velocity <= 0.01700\n---------- X Position <= 0.00000\n----------- MAIN ENGINE\n----------- X Position <= -0.91163\n------------ NOP\n------------ MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angle <= 0.01464\n---------- MAIN ENGINE\n---------- MAIN ENGINE\n-------- Angular Velocity <= -0.02160\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------- Angle <= 0.05061\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n---- RIGHT ENGINE\n-- Angle <= 0.46023\n--- Y Velocity <= -0.01360\n---- MAIN ENGINE\n---- NOP\n--- RIGHT ENGINE"
    config = get_config("lunar_lander")

    trees = [Individual.read_from_string(config, string=tree_str) for _ in range(num_trees)]
    norm_state = True
    print(f"Population has {len(trees)} trees.")

    # for n_jobs in [8, 16, 32]:
    for n_jobs in [-1, 2, 4, 8, 16, 32]:
        title = f"PARALLEL w/ {n_jobs} JOBS" if n_jobs > 1 else "SEQUENTIAL"
        console.rule(title)
        TIME_START = time()
        rl.fill_metrics(config=config, trees=trees,
            alpha=alpha, episodes=episodes, 
            should_norm_state=norm_state,
            n_jobs=n_jobs)
        TIME_END = time()
        print(f"Elapsed time: {TIME_END - TIME_START} seconds")
        rewards = [t.reward for t in trees]
        print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")

    pdb.set_trace()

def run_parallel_experiment_2(episodes=500):
    # Initialize population

    # tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- Car Position <= 2.000\n---- RIGHT\n---- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    # tree_str = "\n- Car Velocity <= -0.02401\n-- LEFT\n-- Car Velocity <= 0.20463\n--- Car Position <= -0.17120\n---- RIGHT\n---- LEFT\n--- RIGHT"
    # tree_str = "\n- Car Velocity <= -0.04286\n-- LEFT\n-- Car Velocity <= 0.07000\n--- Car Velocity <= 0.46567\n---- Car Position <= -0.30667\n----- RIGHT\n----- Car Position <= -0.21000\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"
    # config = get_config("mountain_car")

    tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Angle <= -0.01814\n--- Y Velocity <= -0.04140\n---- Angular Velocity <= -0.03520\n----- LEFT ENGINE\n----- X Velocity <= -0.01060\n------ MAIN ENGINE\n------ Angular Velocity <= 0.02240\n------- LEFT ENGINE\n------- X Velocity <= 0.05960\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n---- NOP\n--- Y Velocity <= 0.25919\n---- X Velocity <= -0.01300\n----- Y Velocity <= -0.05960\n------ Leg 1 is Touching <= -0.16376\n------- MAIN ENGINE\n------- NOP\n------ X Velocity <= -0.02680\n------- RIGHT ENGINE\n------- RIGHT ENGINE\n----- Y Position <= 0.16467\n------ Y Position <= 0.72484\n------- Y Position <= 1.03512\n-------- X Velocity <= 0.79464\n--------- Y Velocity <= -0.02780\n---------- MAIN ENGINE\n---------- NOP\n--------- MAIN ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n------ Y Velocity <= -0.04700\n------- Angle <= 0.07257\n-------- X Position <= -0.01733\n--------- Angular Velocity <= 0.01700\n---------- X Position <= 0.00000\n----------- MAIN ENGINE\n----------- X Position <= -0.91163\n------------ NOP\n------------ MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angle <= 0.01464\n---------- MAIN ENGINE\n---------- MAIN ENGINE\n-------- Angular Velocity <= -0.02160\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------- Angle <= 0.05061\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n---- RIGHT ENGINE\n-- Angle <= 0.46023\n--- Y Velocity <= -0.01360\n---- MAIN ENGINE\n---- NOP\n--- RIGHT ENGINE"
    config = get_config("lunar_lander")

    tree = Individual.read_from_string(config, string=tree_str)
    norm_state = True

    print(f"Evaluating a tree with {tree.get_tree_size()} nodes in {episodes} episodes for task {config['name']}.")

    # for n_jobs in [8, 16, 32]:
    for n_jobs in [-1, 2, 4, 8, 16, 32]:
        title = f"PARALLEL w/ {n_jobs} JOBS" if n_jobs > 1 else "SEQUENTIAL"
        console.rule(title)

        TIME_START = time()
        if n_jobs == -1:
            rewards = rl.collect_rewards(config, tree, episodes,
                should_norm_state=norm_state)
        else:
            rewards = rl.collect_rewards_par(config, tree, episodes, 
                should_norm_state=norm_state, n_jobs=n_jobs)
        TIME_END = time()

        print(f"Elapsed time: {TIME_END - TIME_START} seconds")
        print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")

    pdb.set_trace()

def run_parallel_experiment_3(alpha=0.5, episodes=30, num_trees=100):
    # Initialize population

    # tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- Car Position <= 2.000\n---- RIGHT\n---- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    # tree_str = "\n- Car Velocity <= -0.02401\n-- LEFT\n-- Car Velocity <= 0.20463\n--- Car Position <= -0.17120\n---- RIGHT\n---- LEFT\n--- RIGHT"
    # tree_str = "\n- Car Velocity <= -0.04286\n-- LEFT\n-- Car Velocity <= 0.07000\n--- Car Velocity <= 0.46567\n---- Car Position <= -0.30667\n----- RIGHT\n----- Car Position <= -0.21000\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"
    # config = get_config("mountain_car")

    tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Angle <= -0.01814\n--- Y Velocity <= -0.04140\n---- Angular Velocity <= -0.03520\n----- LEFT ENGINE\n----- X Velocity <= -0.01060\n------ MAIN ENGINE\n------ Angular Velocity <= 0.02240\n------- LEFT ENGINE\n------- X Velocity <= 0.05960\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n---- NOP\n--- Y Velocity <= 0.25919\n---- X Velocity <= -0.01300\n----- Y Velocity <= -0.05960\n------ Leg 1 is Touching <= -0.16376\n------- MAIN ENGINE\n------- NOP\n------ X Velocity <= -0.02680\n------- RIGHT ENGINE\n------- RIGHT ENGINE\n----- Y Position <= 0.16467\n------ Y Position <= 0.72484\n------- Y Position <= 1.03512\n-------- X Velocity <= 0.79464\n--------- Y Velocity <= -0.02780\n---------- MAIN ENGINE\n---------- NOP\n--------- MAIN ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n------ Y Velocity <= -0.04700\n------- Angle <= 0.07257\n-------- X Position <= -0.01733\n--------- Angular Velocity <= 0.01700\n---------- X Position <= 0.00000\n----------- MAIN ENGINE\n----------- X Position <= -0.91163\n------------ NOP\n------------ MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angle <= 0.01464\n---------- MAIN ENGINE\n---------- MAIN ENGINE\n-------- Angular Velocity <= -0.02160\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------- Angle <= 0.05061\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n---- RIGHT ENGINE\n-- Angle <= 0.46023\n--- Y Velocity <= -0.01360\n---- MAIN ENGINE\n---- NOP\n--- RIGHT ENGINE"
    config = get_config("lunar_lander")

    trees = [Individual.read_from_string(config, string=tree_str) for _ in range(num_trees)]
    norm_state = True
    print(f"Population has {len(trees)} trees.")

    # for n_jobs in [8, 16, 32]:
    for n_jobs in [32, 16, 8, 4, 2, -1]:
    # for n_jobs in [-1, 2, 4, 8, 16, 32]:
        title = f"PARALLEL w/ {n_jobs} JOBS" if n_jobs > 1 else "SEQUENTIAL"
        console.rule(title)
        print(f"[yellow]Parallel trees with sequential episodes:")
        TIME_START = time()
        
        rl.fill_metrics(config=config, trees=trees,
            alpha=alpha, episodes=episodes, 
            should_norm_state=norm_state,
            n_jobs=n_jobs)
        
        TIME_END = time()
        print(f"Elapsed time: {TIME_END - TIME_START} seconds")
        rewards = [t.reward for t in trees]
        print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")
        print()
        
        print(f"[yellow]Parallel episodes with sequential trees:")
        TIME_START = time()
        
        rl.collect_metrics(config, trees, alpha, episodes=episodes,
            should_norm_state=norm_state, penalize_std=True, 
            should_fill_attributes=True, render=False, n_jobs=n_jobs)
        
        TIME_END = time()
        print(f"Elapsed time: {TIME_END - TIME_START} seconds")
        rewards = [t.reward for t in trees]
        print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")

    pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Experiments')
    parser.add_argument('-e','--experiment', help="Which experiment to run?", type=int, required=True)
    args = vars(parser.parse_args())

    print(f"Starting experiment {args['experiment']}. . .")
    
    if args["experiment"] == 1:
        run_parallel_experiment_1()
    elif args["experiment"] == 2:
        run_parallel_experiment_2()
    elif args["experiment"] == 3:
        run_parallel_experiment_3()
