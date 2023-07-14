import argparse
import pdb
import numpy as np
from rich import print

import erltrees.rl.utils as rl
from erltrees.il.dataset_creation import get_model

from erltrees.rl.configs import get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize tree')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    parser.add_argument('-c','--class', help='Tree is QTree, Distilled Tree, or Viztree?', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=-1, type=int)
    parser.add_argument('--grading_episodes', help='How many episodes should we use to measure model\'s accuracy?', required=False, default=100, type=int)
    parser.add_argument('--should_print_state', help='Should print state?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_render', help='Should render model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # Initialization    
    config = get_config(args['task'])
    model = get_model(args['class'], args['filename'], config)

    # Evaluating model
    rewards = [rl.collect_metrics(config, trees=[model], episodes=1)[0][0] for _ in range(args['iterations'])]
    print(f"Average reward is {np.mean(rewards)} ± {np.std(rewards)}.")
    
    if args['task_solution_threshold'] != -1:
        success_rate = np.mean([1 if r > args['task_solution_threshold'] else 0 for r in rewards])
        print(f"Success rate is {success_rate}")

    # Printing size
    tree_size = None
    if args['class'] == "ClassificationTree":
        tree_size = model.get_size()
        depth = model.model.get_depth()
        print(f"Tree has {tree_size} nodes and depth {depth}.")

    # Rendering model
    if args['should_render']:
        rl.collect_metrics(config, [model], episodes=10,
            should_norm_state=False, should_fill_attributes=True, 
            render=True, verbose=True)