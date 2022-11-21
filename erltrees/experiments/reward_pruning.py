import argparse
from datetime import datetime
import pdb
import gym
import time
import numpy as np
from rich import print
from erltrees.evo.evo_tree import Individual
import erltrees.rl.configs as configs
import erltrees.rl.utils as rl
from scipy.stats import ks_2samp
from erltrees.io import printv, console

def save_history_to_file(filepath, trees, elapsed_time, prefix):
    string = prefix

    rewards = [tree.reward for tree in trees]
    sizes = [tree.get_tree_size() for tree in trees]
    success_rates = [tree.success_rate for tree in trees]

    string += f"Mean Best Reward: {np.mean(rewards)} +- {np.std(rewards)}\n"
    string += f"Mean Best Size: {np.mean(sizes)}\n"
    string += f"Average Evaluations to Success: -------\n"
    string += f"Success Rate: {np.mean(success_rates)}\n"
    if elapsed_time:
        string += f"Elapsed time: {elapsed_time} seconds"
    string += "\n-----\n\n"

    for i, tree in enumerate(trees):
        string += f"Tree #{i} (Reward: {tree.reward} +- {tree.std_reward}, Success Rate: {tree.success_rate}, Size: {tree.get_tree_size()})\n"
        string += "----------\n"
        string += str(tree)
        string += "\n"

    with open(filepath, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def reward_pruning(tree, node, config, episodes=100, alpha=0,
    task_solution_threshold=0, should_norm_state=True, 
    should_use_pvalue=True, n_jobs=4, verbose=False):

    tree = tree.copy()
    
    nodes = tree.get_node_list(get_inners=True, get_leaves=False)
    # nodes = filter(lambda x : x.get_tree_size() == 3, nodes)
    nodes.sort(key=lambda x : x.get_tree_size())
    node_paths = [node.get_path() for node in nodes]
    node_paths = node_paths[:-1] #shouldn't try to remove root

    rewards_curr = rl.collect_rewards_par(config, tree, episodes, should_norm_state, n_jobs=n_jobs)
    tree.fitness = rl.calc_fitness(np.mean(rewards_curr), np.std(rewards_curr), tree.get_tree_size(), alpha, should_penalize_std=True)
    tree.success_rate = np.mean([(1 if r > task_solution_threshold else 0) for r in rewards_curr])

    for node_path in node_paths:
        printv("-----------------------", verbose)
        printv(f"-- Pruning a tree with {tree.get_tree_size()} nodes.", verbose)

        tree_alt_1 = tree.copy()
        node = tree_alt_1.get_node_by_path(node_path)
        node.left.cut_parent()

        rewards_alt_1 = rl.collect_rewards_par(config, tree_alt_1, episodes, should_norm_state, n_jobs=n_jobs)
        tree_alt_1.fitness = rl.calc_fitness(np.mean(rewards_alt_1), np.std(rewards_alt_1), tree_alt_1.get_tree_size(), alpha, should_penalize_std=True)
        tree_alt_1.success_rate = np.mean([(1 if r > task_solution_threshold else 0) for r in rewards_alt_1])

        printv(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its left node.", verbose)
        printv(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}. (size: {tree.get_tree_size()}, fit: {tree.fitness}, sr: {tree.success_rate})", verbose)
        printv(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_1))} +- {'{:.3f}'.format(np.std(rewards_alt_1))}. (size: {tree_alt_1.get_tree_size()}, fit: {tree_alt_1.fitness}, sr: {tree_alt_1.success_rate})", verbose)

        stats, pvalue = ks_2samp(rewards_curr, rewards_alt_1)
        printv(f"------ KL Stat: {stats}, P-value: {pvalue}", verbose)

        if (tree_alt_1.fitness > tree.fitness) or (tree_alt_1.success_rate > tree.success_rate) or (pvalue > 0.9 and should_use_pvalue):
            printv(f"------ [green]Maintaining change.[/green]", verbose)
            tree = tree_alt_1
            rewards_curr = rewards_alt_1
        else:
            printv(f"------ [red]Undoing change.[/red]", verbose)

            tree_alt_2 = tree.copy()
            node = tree_alt_2.get_node_by_path(node_path)
            node.right.cut_parent()

            rewards_alt_2 = rl.collect_rewards_par(config, tree_alt_2, episodes, should_norm_state, n_jobs=n_jobs)
            tree_alt_2.fitness = rl.calc_fitness(np.mean(rewards_alt_2), np.std(rewards_alt_2), tree_alt_2.get_tree_size(), alpha, should_penalize_std=True)
            tree_alt_2.success_rate = np.mean([(1 if r > task_solution_threshold else 0) for r in rewards_alt_2])

            printv(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its right node.", verbose)
            printv(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}. (size: {tree.get_tree_size()}, fit: {tree.fitness}, sr: {tree.success_rate})", verbose)
            printv(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_2))} +- {'{:.3f}'.format(np.std(rewards_alt_2))}. (size: {tree_alt_2.get_tree_size()}, fit: {tree_alt_2.fitness}, sr: {tree_alt_2.success_rate})", verbose)
            
            stats, pvalue = ks_2samp(rewards_curr, rewards_alt_2)
            printv(f"------ KL Stat: {stats}, P-value: {pvalue}", verbose)

            if (tree_alt_2.fitness > tree.fitness) or (tree_alt_2.success_rate > tree.success_rate) or (pvalue > 0.9 and should_use_pvalue):
                printv(f"------ [green]Maintaining change.[/green]", verbose)
                tree = tree_alt_2
                rewards_curr = rewards_alt_2
            else:
                printv(f"------ [red]Undoing change.[/red]", verbose)

    return tree        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Pruning')
    parser.add_argument('--n_jobs', help="How many jobs to run?", type=int, required=False, default=-1)
    parser.add_argument('--alpha', help='Which alpha to use?', required=True, default=1.0, type=float)
    parser.add_argument('--rounds', help='How many rounds for reward pruning?', required=True, default=1, type=int)
    parser.add_argument('--simulations', help='How many simulations to run?', required=True, type=int)
    parser.add_argument('--episodes', help='How many episodes to run?', required=True, type=int)
    parser.add_argument('--should_use_pvalue', help='Should use p-value to detect if trees are equal?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=True, default=None, type=int)
    args = vars(parser.parse_args())

    filepath = "data/reward_pruning_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt"

    # tree_str = "\n- Y Position <= 0.001\n-- Angle <= 0.153\n--- Angle <= -0.182\n---- Angular Velocity <= 0.000\n----- LEFT ENGINE\n----- Y Position <= -0.032\n------ LEFT ENGINE\n------ NOP\n---- Leg 1 is Touching <= 0.500\n----- Leg 2 is Touching <= 0.500\n------ X Velocity <= -0.001\n------- NOP\n------- LEFT ENGINE\n------ Angular Velocity <= -0.135\n------- LEFT ENGINE\n------- NOP\n----- Angular Velocity <= 0.252\n------ NOP\n------ MAIN ENGINE\n--- X Velocity <= 0.001\n---- Angle <= 0.319\n----- X Velocity <= -0.002\n------ NOP\n------ Y Velocity <= -0.000\n------- RIGHT ENGINE\n------- NOP\n----- RIGHT ENGINE\n---- X Position <= -0.301\n----- RIGHT ENGINE\n----- NOP\n-- Y Velocity <= -0.091\n--- X Position <= 0.321\n---- Y Velocity <= -0.244\n----- Y Position <= 1.310\n------ Y Position <= 0.674\n------- Angle <= -0.082\n-------- MAIN ENGINE\n-------- Angular Velocity <= 0.070\n--------- MAIN ENGINE\n--------- MAIN ENGINE\n------- Angle <= 0.091\n-------- Angular Velocity <= -0.238\n--------- Angle <= -0.223\n---------- LEFT ENGINE\n---------- NOP\n--------- Angle <= -0.146\n---------- MAIN ENGINE\n---------- Angular Velocity <= -0.031\n----------- Y Position <= 1.181\n------------ Angle <= 0.004\n------------- MAIN ENGINE\n------------- MAIN ENGINE\n------------ NOP\n----------- Y Velocity <= -0.291\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n-------- Angular Velocity <= -0.099\n--------- MAIN ENGINE\n--------- MAIN ENGINE\n------ X Velocity <= -0.077\n------- Angle <= -0.010\n-------- NOP\n-------- X Velocity <= -0.292\n--------- RIGHT ENGINE\n--------- NOP\n------- Angle <= 0.046\n-------- X Velocity <= 0.354\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n-------- MAIN ENGINE\n----- Y Position <= 0.213\n------ Y Velocity <= -0.104\n------- Y Position <= 0.126\n-------- Leg 2 is Touching <= 0.500\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n-------- Y Velocity <= -0.194\n--------- MAIN ENGINE\n--------- Y Position <= 0.166\n---------- Y Velocity <= -0.132\n----------- MAIN ENGINE\n----------- MAIN ENGINE\n---------- MAIN ENGINE\n------- Angle <= 0.028\n-------- Angular Velocity <= 0.016\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n-------- Y Position <= 0.096\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------ X Velocity <= -0.157\n------- Angle <= -0.020\n-------- Y Position <= 1.318\n--------- RIGHT ENGINE\n--------- NOP\n-------- RIGHT ENGINE\n------- Y Velocity <= -0.210\n-------- Y Position <= 0.534\n--------- Angular Velocity <= 0.067\n---------- X Position <= 0.056\n----------- Y Velocity <= -0.223\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n----------- MAIN ENGINE\n---------- MAIN ENGINE\n--------- Angle <= 0.131\n---------- X Velocity <= 0.194\n----------- LEFT ENGINE\n----------- LEFT ENGINE\n---------- MAIN ENGINE\n-------- Angle <= 0.136\n--------- X Velocity <= 0.161\n---------- Angle <= -0.059\n----------- LEFT ENGINE\n----------- X Velocity <= -0.009\n------------ RIGHT ENGINE\n------------ Y Position <= 0.282\n------------- LEFT ENGINE\n------------- LEFT ENGINE\n---------- LEFT ENGINE\n--------- RIGHT ENGINE\n---- Angle <= 0.092\n----- LEFT ENGINE\n----- LEFT ENGINE\n--- X Velocity <= 0.151\n---- Leg 1 is Touching <= 0.500\n----- Angle <= 0.060\n------ X Velocity <= -0.130\n------- Angle <= -0.037\n-------- RIGHT ENGINE\n-------- RIGHT ENGINE\n------- Angular Velocity <= 0.023\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n------ Y Velocity <= -0.072\n------- X Velocity <= -0.041\n-------- RIGHT ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n----- X Position <= -0.574\n------ RIGHT ENGINE\n------ NOP\n---- X Position <= -0.494\n----- MAIN ENGINE\n----- Angle <= 0.037\n------ LEFT ENGINE\n------ LEFT ENGINE"
    # tree_str = "\n- Leg 1 is Touching <= 0.00000\n-- Angle <= 0.00064\n--- Y Velocity <= -0.02000\n---- Y Position <= 0.78800\n----- Y Velocity <= -0.06020\n------ Angular Velocity <= -0.01460\n------- Y Position <= 0.38000\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n------- Y Velocity <= -0.08040\n-------- MAIN ENGINE\n-------- X Velocity <= -0.03400\n--------- RIGHT ENGINE\n--------- MAIN ENGINE\n------ Y Position <= 0.11000\n------- X Velocity <= 0.02300\n-------- MAIN ENGINE\n-------- Y Velocity <= -0.03660\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n------- Angle <= -0.03342\n-------- LEFT ENGINE\n-------- Y Position <= 0.42133\n--------- Angular Velocity <= -0.01620\n---------- NOP\n---------- RIGHT ENGINE\n--------- LEFT ENGINE\n----- X Velocity <= -0.02540\n------ NOP\n------ LEFT ENGINE\n---- Angle <= -0.02324\n----- LEFT ENGINE\n----- X Position <= -0.00933\n------ NOP\n------ LEFT ENGINE\n--- Y Velocity <= -0.06460\n---- MAIN ENGINE\n---- X Velocity <= -0.01680\n----- RIGHT ENGINE\n----- Y Velocity <= -0.01560\n------ Y Position <= 0.32885\n------- MAIN ENGINE\n------- Angle <= 0.07257\n-------- X Velocity <= 0.02300\n--------- Angular Velocity <= 0.01540\n---------- Angular Velocity <= -0.02220\n----------- LEFT ENGINE\n----------- LEFT ENGINE\n---------- RIGHT ENGINE\n--------- X Position <= 0.09200\n---------- Angle <= 0.03056\n----------- LEFT ENGINE\n----------- MAIN ENGINE\n---------- LEFT ENGINE\n-------- RIGHT ENGINE\n------ Angle <= 0.02196\n------- X Position <= 0.00000\n-------- RIGHT ENGINE\n-------- NOP\n------- RIGHT ENGINE\n-- Angle <= -0.06016\n--- LEFT ENGINE\n--- Y Velocity <= -0.01140\n---- MAIN ENGINE\n---- NOP"
    tree_str = "\n- Leg 1 is Touching <= 0.500\n-- Y Velocity <= -0.092\n--- Angle <= -0.045\n---- Y Position <= 1.188\n----- Y Velocity <= -0.290\n------ Angular Velocity <= -0.160\n------- LEFT ENGINE\n------- X Velocity <= 0.074\n-------- Y Position <= 0.578\n--------- MAIN ENGINE\n--------- Angular Velocity <= -0.025\n---------- X Velocity <= -0.195\n----------- Angle <= -0.145\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n----------- LEFT ENGINE\n---------- X Velocity <= -0.076\n----------- MAIN ENGINE\n----------- Angular Velocity <= 0.077\n------------ LEFT ENGINE\n------------ MAIN ENGINE\n-------- Y Position <= 0.372\n--------- MAIN ENGINE\n--------- Angular Velocity <= 0.249\n---------- LEFT ENGINE\n---------- MAIN ENGINE\n------ Angle <= -0.133\n------- Y Position <= 0.144\n-------- Y Velocity <= -0.165\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n-------- X Position <= -0.280\n--------- LEFT ENGINE\n--------- LEFT ENGINE\n------- Y Position <= 0.180\n-------- Y Velocity <= -0.164\n--------- MAIN ENGINE\n--------- X Velocity <= 0.060\n---------- X Position <= -0.056\n----------- MAIN ENGINE\n----------- LEFT ENGINE\n---------- LEFT ENGINE\n-------- Y Velocity <= -0.219\n--------- Angular Velocity <= -0.070\n---------- LEFT ENGINE\n---------- X Position <= -0.203\n----------- RIGHT ENGINE\n----------- Y Position <= 0.624\n------------ X Velocity <= 0.028\n------------- X Position <= 0.076\n-------------- MAIN ENGINE\n-------------- LEFT ENGINE\n------------- LEFT ENGINE\n------------ LEFT ENGINE\n--------- X Velocity <= -0.035\n---------- X Position <= -0.018\n----------- RIGHT ENGINE\n----------- LEFT ENGINE\n---------- X Position <= -0.114\n----------- LEFT ENGINE\n----------- LEFT ENGINE\n----- Angular Velocity <= -0.048\n------ X Velocity <= -0.419\n------- Angle <= -0.222\n-------- LEFT ENGINE\n-------- NOP\n------- LEFT ENGINE\n------ X Velocity <= -0.224\n------- X Velocity <= -0.492\n-------- RIGHT ENGINE\n-------- NOP\n------- LEFT ENGINE\n---- Y Position <= 0.126\n----- Y Velocity <= -0.122\n------ Angle <= 0.273\n------- Y Velocity <= -0.165\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.343\n--------- LEFT ENGINE\n--------- X Velocity <= 0.148\n---------- MAIN ENGINE\n---------- MAIN ENGINE\n------- MAIN ENGINE\n------ Angular Velocity <= -0.324\n------- LEFT ENGINE\n------- Angle <= 0.021\n-------- X Velocity <= 0.094\n--------- X Position <= 0.031\n---------- Angular Velocity <= 0.060\n----------- Y Velocity <= -0.106\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n----------- RIGHT ENGINE\n---------- LEFT ENGINE\n--------- LEFT ENGINE\n-------- Y Position <= 0.077\n--------- X Velocity <= -0.086\n---------- RIGHT ENGINE\n---------- MAIN ENGINE\n--------- RIGHT ENGINE\n----- Y Velocity <= -0.295\n------ X Velocity <= -0.136\n------- X Velocity <= -0.361\n-------- RIGHT ENGINE\n-------- Y Position <= 0.816\n--------- Y Velocity <= -0.399\n---------- MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angular Velocity <= 0.033\n---------- Angle <= 0.057\n----------- Y Velocity <= -0.636\n------------ MAIN ENGINE\n------------ NOP\n----------- RIGHT ENGINE\n---------- RIGHT ENGINE\n------- Angle <= 0.194\n-------- Y Position <= 0.726\n--------- Y Position <= 0.374\n---------- MAIN ENGINE\n---------- X Position <= -0.087\n----------- Y Velocity <= -0.411\n------------ MAIN ENGINE\n------------ RIGHT ENGINE\n----------- Angular Velocity <= -0.141\n------------ LEFT ENGINE\n------------ Angle <= 0.019\n------------- MAIN ENGINE\n------------- Y Velocity <= -0.443\n-------------- MAIN ENGINE\n-------------- MAIN ENGINE\n--------- Angular Velocity <= -0.122\n---------- LEFT ENGINE\n---------- X Velocity <= 0.338\n----------- Y Velocity <= -0.406\n------------ MAIN ENGINE\n------------ X Velocity <= -0.071\n------------- Angle <= 0.064\n-------------- Angular Velocity <= -0.042\n--------------- NOP\n--------------- Y Position <= 1.047\n---------------- MAIN ENGINE\n---------------- NOP\n-------------- RIGHT ENGINE\n------------- Angle <= 0.058\n-------------- Y Position <= 1.028\n--------------- MAIN ENGINE\n--------------- Angular Velocity <= 0.031\n---------------- LEFT ENGINE\n---------------- MAIN ENGINE\n-------------- X Velocity <= -0.016\n--------------- Angular Velocity <= 0.036\n---------------- MAIN ENGINE\n---------------- RIGHT ENGINE\n--------------- MAIN ENGINE\n----------- Angular Velocity <= 0.218\n------------ LEFT ENGINE\n------------ MAIN ENGINE\n-------- X Velocity <= 0.047\n--------- Y Velocity <= -0.487\n---------- MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angle <= 0.455\n---------- Y Velocity <= -0.395\n----------- MAIN ENGINE\n----------- MAIN ENGINE\n---------- RIGHT ENGINE\n------ X Velocity <= -0.062\n------- Angle <= 0.001\n-------- Y Position <= 0.666\n--------- X Position <= 0.016\n---------- RIGHT ENGINE\n---------- RIGHT ENGINE\n--------- X Velocity <= -0.132\n---------- RIGHT ENGINE\n---------- LEFT ENGINE\n-------- RIGHT ENGINE\n------- Angular Velocity <= 0.088\n-------- Y Position <= 0.491\n--------- Y Velocity <= -0.233\n---------- Y Position <= 0.279\n----------- MAIN ENGINE\n----------- X Position <= 0.074\n------------ X Velocity <= 0.034\n------------- Angle <= -0.016\n-------------- MAIN ENGINE\n-------------- RIGHT ENGINE\n------------- MAIN ENGINE\n------------ LEFT ENGINE\n---------- Y Position <= 0.227\n----------- Y Velocity <= -0.147\n------------ MAIN ENGINE\n------------ RIGHT ENGINE\n----------- X Velocity <= 0.056\n------------ RIGHT ENGINE\n------------ Angle <= 0.026\n------------- LEFT ENGINE\n------------- RIGHT ENGINE\n--------- Angle <= 0.122\n---------- X Velocity <= 0.013\n----------- Angular Velocity <= -0.058\n------------ LEFT ENGINE\n------------ Angle <= 0.033\n------------- Y Position <= 0.659\n-------------- MAIN ENGINE\n-------------- LEFT ENGINE\n------------- RIGHT ENGINE\n----------- X Position <= -0.150\n------------ LEFT ENGINE\n------------ LEFT ENGINE\n---------- X Velocity <= 0.045\n----------- RIGHT ENGINE\n----------- Angle <= 0.285\n------------ X Velocity <= 0.264\n------------- Angular Velocity <= -0.174\n-------------- LEFT ENGINE\n-------------- Y Velocity <= -0.196\n--------------- MAIN ENGINE\n--------------- RIGHT ENGINE\n------------- LEFT ENGINE\n------------ X Velocity <= 0.195\n------------- RIGHT ENGINE\n------------- MAIN ENGINE\n-------- X Velocity <= 0.441\n--------- Angular Velocity <= 0.186\n---------- X Velocity <= 0.188\n----------- RIGHT ENGINE\n----------- Angle <= 0.145\n------------ LEFT ENGINE\n------------ RIGHT ENGINE\n---------- RIGHT ENGINE\n--------- Angular Velocity <= 0.352\n---------- LEFT ENGINE\n---------- RIGHT ENGINE\n--- Y Position <= 0.001\n---- X Position <= -0.114\n----- Angle <= 0.218\n------ Leg 2 is Touching <= 0.500\n------- RIGHT ENGINE\n------- NOP\n------ RIGHT ENGINE\n----- X Position <= 0.120\n------ Leg 2 is Touching <= 0.500\n------- X Position <= 0.040\n-------- X Position <= -0.032\n--------- RIGHT ENGINE\n--------- NOP\n-------- X Velocity <= -0.001\n--------- X Position <= 0.044\n---------- NOP\n---------- X Velocity <= -0.033\n----------- NOP\n----------- LEFT ENGINE\n--------- LEFT ENGINE\n------- Angular Velocity <= -0.178\n-------- LEFT ENGINE\n-------- NOP\n------ LEFT ENGINE\n---- X Velocity <= -0.076\n----- Angle <= 0.002\n------ X Velocity <= -0.340\n------- RIGHT ENGINE\n------- LEFT ENGINE\n------ RIGHT ENGINE\n----- Angle <= 0.022\n------ Y Position <= 0.689\n------- Angle <= -0.073\n-------- LEFT ENGINE\n-------- LEFT ENGINE\n------- LEFT ENGINE\n------ Angular Velocity <= 0.005\n------- X Position <= -0.013\n-------- RIGHT ENGINE\n-------- LEFT ENGINE\n------- Angle <= 0.083\n-------- X Velocity <= 0.094\n--------- RIGHT ENGINE\n--------- LEFT ENGINE\n-------- Y Position <= 0.064\n--------- Y Velocity <= -0.070\n---------- MAIN ENGINE\n---------- RIGHT ENGINE\n--------- RIGHT ENGINE\n-- Angle <= 0.187\n--- Angle <= -0.177\n---- X Position <= -0.315\n----- NOP\n----- Angular Velocity <= 0.273\n------ Angle <= -0.192\n------- LEFT ENGINE\n------- X Position <= 0.244\n-------- NOP\n-------- LEFT ENGINE\n------ MAIN ENGINE\n---- Y Velocity <= -0.060\n----- Leg 2 is Touching <= 0.500\n------ MAIN ENGINE\n------ Angular Velocity <= -0.371\n------- LEFT ENGINE\n------- NOP\n----- X Velocity <= 0.098\n------ X Position <= -0.467\n------- MAIN ENGINE\n------- Angular Velocity <= 0.313\n-------- X Position <= 0.247\n--------- NOP\n--------- Angle <= -0.161\n---------- LEFT ENGINE\n---------- NOP\n-------- RIGHT ENGINE\n------ Leg 2 is Touching <= 0.500\n------- LEFT ENGINE\n------- Angle <= -0.134\n-------- LEFT ENGINE\n-------- NOP\n--- Angular Velocity <= -0.000\n---- Angle <= 0.328\n----- X Position <= -0.298\n------ RIGHT ENGINE\n------ Angle <= 0.323\n------- NOP\n------- X Position <= -0.267\n-------- RIGHT ENGINE\n-------- NOP\n----- RIGHT ENGINE\n---- Angle <= 0.312\n----- Y Velocity <= -0.004\n------ Angle <= 0.232\n------- X Velocity <= -0.099\n-------- RIGHT ENGINE\n-------- NOP\n------- RIGHT ENGINE\n------ X Position <= -0.380\n------- RIGHT ENGINE\n------- NOP\n----- X Position <= -0.254\n------ Angle <= 0.321\n------- X Position <= -0.276\n-------- RIGHT ENGINE\n-------- NOP\n------- RIGHT ENGINE\n------ Angle <= 0.329\n------- NOP\n------- RIGHT ENGINE"
    config = configs.get_config("lunar_lander")
    # tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    # config = configs.get_config("mountain_car")
    original_tree = Individual.read_from_string(config, tree_str)
    # tree.denormalize_thresholds()

    command_line = str(args)
    command_line += "\n\npython -m erltrees.experiments.reward_pruning " + " ".join([f"--{key} {val}" for (key, val) in args.items()])
    command_line += "\n\n"
    command_line += "Original tree: \n"
    command_line += str(original_tree)
    command_line += "\n\n"

    final_trees = []
    
    START_TIME = time.time()
    for _ in range(args["simulations"]):
        tree = original_tree.copy()

        for round in range(args["rounds"]):
            console.rule(f"Round {round + 1} / {args['rounds']}")
            
            tree = reward_pruning(tree, tree, config, episodes=args["episodes"], alpha=args["alpha"], 
                should_norm_state=False, should_use_pvalue=args["should_use_pvalue"], n_jobs=args["n_jobs"],
                task_solution_threshold=args["task_solution_threshold"], verbose=True)

        rl.collect_metrics(config, [tree], args["alpha"], 1000, should_norm_state=False, penalize_std=True, 
            should_fill_attributes=True, task_solution_threshold=args["task_solution_threshold"], n_jobs=args["n_jobs"])

        final_trees.append(tree)

        save_history_to_file(filepath, final_trees, time.time() - START_TIME, command_line)

    END_TIME = time.time()
    elapsed_time = END_TIME - START_TIME
    print(f"Elapsed time: {elapsed_time} seconds.")