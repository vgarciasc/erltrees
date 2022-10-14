import pdb
import gym
import numpy as np
from rich import print
from erltrees.evo.evo_tree import Individual
import erltrees.rl.configs as configs
import erltrees.rl.utils as rl
from scipy.stats import ks_2samp

def reward_pruning(tree, node, config, episodes=100, alpha=0, should_norm_state=True, n_jobs=4):
    nodes = tree.get_node_list(get_inner=True, get_leaf=False)
    # nodes = filter(lambda x : x.get_tree_size() == 3, nodes)
    nodes.sort(key=lambda x : x.get_tree_size())
    node_paths = [node.get_path() for node in nodes]

    rewards_curr = rl.collect_rewards(config, tree, episodes, should_norm_state)
    rewards_curr = [r - alpha * tree.get_tree_size() for r in rewards_curr]

    for node_path in node_paths:
        print("-----------------------")
        print(f"-- Pruning a tree with {tree.get_tree_size()} nodes.")

        tree_alt_1 = tree.copy()
        node = tree_alt_1.get_node_by_path(node_path)
        node.left.cut_parent()

        rewards_alt_1 = rl.collect_rewards(config, tree_alt_1, episodes, should_norm_state)
        rewards_alt_1 = [r - alpha * tree.get_tree_size() for r in rewards_alt_1]

        print(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its left node.")
        print(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}.")
        print(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_1))} +- {'{:.3f}'.format(np.std(rewards_alt_1))}.")

        stats, pvalue = ks_2samp(rewards_curr, rewards_alt_1)
        print(f"------ KL Stat: {stats}, P-value: {pvalue}")

        if (pvalue < 0.05 and tree_alt_1.fitness > tree.fitness) or \
            (pvalue > 0.75 and tree_alt_1.get_tree_size() < tree.get_tree_size()):

            print(f"------ [green]Maintaining change.[/green]")
            tree = tree_alt_1
            rewards_curr = rewards_alt_1
        else:
            print(f"------ [red]Undoing change.[/red]")

            tree_alt_2 = tree.copy()
            node = tree_alt_2.get_node_by_path(node_path)
            node.right.cut_parent()

            rewards_alt_2 = rl.collect_rewards(config, tree_alt_2, episodes, should_norm_state)
            rewards_alt_2 = [r - alpha * tree.get_tree_size() for r in rewards_alt_2]

            print(f"---- Replaced '{config['attributes'][node.attribute][0]} <= {node.threshold}' with its right node.")
            print(f"------ Tree:     {'{:.3f}'.format(np.mean(rewards_curr))} +- {'{:.3f}'.format(np.std(rewards_curr))}.")
            print(f"------ Alt tree: {'{:.3f}'.format(np.mean(rewards_alt_2))} +- {'{:.3f}'.format(np.std(rewards_alt_2))}.")
            
            stats, pvalue = ks_2samp(rewards_curr, rewards_alt_2)
            print(f"------ KL Stat: {stats}, P-value: {pvalue}")

            if (pvalue < 0.05 and tree_alt_2.fitness > tree.fitness) or \
                (pvalue > 0.75 and tree_alt_2.get_tree_size() < tree.get_tree_size()):

                print(f"------ [green]Maintaining change.[/green]")
                tree = tree_alt_2
                rewards_curr = rewards_alt_2
            else:
                print(f"------ [red]Undoing change.[/red]")

    return tree        

if __name__ == "__main__":
    # tree_str = "\n- Y Position <= 0.001\n-- Angle <= 0.153\n--- Angle <= -0.182\n---- Angular Velocity <= 0.000\n----- LEFT ENGINE\n----- Y Position <= -0.032\n------ LEFT ENGINE\n------ NOP\n---- Leg 1 is Touching <= 0.500\n----- Leg 2 is Touching <= 0.500\n------ X Velocity <= -0.001\n------- NOP\n------- LEFT ENGINE\n------ Angular Velocity <= -0.135\n------- LEFT ENGINE\n------- NOP\n----- Angular Velocity <= 0.252\n------ NOP\n------ MAIN ENGINE\n--- X Velocity <= 0.001\n---- Angle <= 0.319\n----- X Velocity <= -0.002\n------ NOP\n------ Y Velocity <= -0.000\n------- RIGHT ENGINE\n------- NOP\n----- RIGHT ENGINE\n---- X Position <= -0.301\n----- RIGHT ENGINE\n----- NOP\n-- Y Velocity <= -0.091\n--- X Position <= 0.321\n---- Y Velocity <= -0.244\n----- Y Position <= 1.310\n------ Y Position <= 0.674\n------- Angle <= -0.082\n-------- MAIN ENGINE\n-------- Angular Velocity <= 0.070\n--------- MAIN ENGINE\n--------- MAIN ENGINE\n------- Angle <= 0.091\n-------- Angular Velocity <= -0.238\n--------- Angle <= -0.223\n---------- LEFT ENGINE\n---------- NOP\n--------- Angle <= -0.146\n---------- MAIN ENGINE\n---------- Angular Velocity <= -0.031\n----------- Y Position <= 1.181\n------------ Angle <= 0.004\n------------- MAIN ENGINE\n------------- MAIN ENGINE\n------------ NOP\n----------- Y Velocity <= -0.291\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n-------- Angular Velocity <= -0.099\n--------- MAIN ENGINE\n--------- MAIN ENGINE\n------ X Velocity <= -0.077\n------- Angle <= -0.010\n-------- NOP\n-------- X Velocity <= -0.292\n--------- RIGHT ENGINE\n--------- NOP\n------- Angle <= 0.046\n-------- X Velocity <= 0.354\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n-------- MAIN ENGINE\n----- Y Position <= 0.213\n------ Y Velocity <= -0.104\n------- Y Position <= 0.126\n-------- Leg 2 is Touching <= 0.500\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n-------- Y Velocity <= -0.194\n--------- MAIN ENGINE\n--------- Y Position <= 0.166\n---------- Y Velocity <= -0.132\n----------- MAIN ENGINE\n----------- MAIN ENGINE\n---------- MAIN ENGINE\n------- Angle <= 0.028\n-------- Angular Velocity <= 0.016\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n-------- Y Position <= 0.096\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------ X Velocity <= -0.157\n------- Angle <= -0.020\n-------- Y Position <= 1.318\n--------- RIGHT ENGINE\n--------- NOP\n-------- RIGHT ENGINE\n------- Y Velocity <= -0.210\n-------- Y Position <= 0.534\n--------- Angular Velocity <= 0.067\n---------- X Position <= 0.056\n----------- Y Velocity <= -0.223\n------------ MAIN ENGINE\n------------ MAIN ENGINE\n----------- MAIN ENGINE\n---------- MAIN ENGINE\n--------- Angle <= 0.131\n---------- X Velocity <= 0.194\n----------- LEFT ENGINE\n----------- LEFT ENGINE\n---------- MAIN ENGINE\n-------- Angle <= 0.136\n--------- X Velocity <= 0.161\n---------- Angle <= -0.059\n----------- LEFT ENGINE\n----------- X Velocity <= -0.009\n------------ RIGHT ENGINE\n------------ Y Position <= 0.282\n------------- LEFT ENGINE\n------------- LEFT ENGINE\n---------- LEFT ENGINE\n--------- RIGHT ENGINE\n---- Angle <= 0.092\n----- LEFT ENGINE\n----- LEFT ENGINE\n--- X Velocity <= 0.151\n---- Leg 1 is Touching <= 0.500\n----- Angle <= 0.060\n------ X Velocity <= -0.130\n------- Angle <= -0.037\n-------- RIGHT ENGINE\n-------- RIGHT ENGINE\n------- Angular Velocity <= 0.023\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n------ Y Velocity <= -0.072\n------- X Velocity <= -0.041\n-------- RIGHT ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n----- X Position <= -0.574\n------ RIGHT ENGINE\n------ NOP\n---- X Position <= -0.494\n----- MAIN ENGINE\n----- Angle <= 0.037\n------ LEFT ENGINE\n------ LEFT ENGINE"
    tree_str = "\n- Leg 1 is Touching <= 0.00000\n-- Angle <= 0.00064\n--- Y Velocity <= -0.02000\n---- Y Position <= 0.78800\n----- Y Velocity <= -0.06020\n------ Angular Velocity <= -0.01460\n------- Y Position <= 0.38000\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n------- Y Velocity <= -0.08040\n-------- MAIN ENGINE\n-------- X Velocity <= -0.03400\n--------- RIGHT ENGINE\n--------- MAIN ENGINE\n------ Y Position <= 0.11000\n------- X Velocity <= 0.02300\n-------- MAIN ENGINE\n-------- Y Velocity <= -0.03660\n--------- MAIN ENGINE\n--------- LEFT ENGINE\n------- Angle <= -0.03342\n-------- LEFT ENGINE\n-------- Y Position <= 0.42133\n--------- Angular Velocity <= -0.01620\n---------- NOP\n---------- RIGHT ENGINE\n--------- LEFT ENGINE\n----- X Velocity <= -0.02540\n------ NOP\n------ LEFT ENGINE\n---- Angle <= -0.02324\n----- LEFT ENGINE\n----- X Position <= -0.00933\n------ NOP\n------ LEFT ENGINE\n--- Y Velocity <= -0.06460\n---- MAIN ENGINE\n---- X Velocity <= -0.01680\n----- RIGHT ENGINE\n----- Y Velocity <= -0.01560\n------ Y Position <= 0.32885\n------- MAIN ENGINE\n------- Angle <= 0.07257\n-------- X Velocity <= 0.02300\n--------- Angular Velocity <= 0.01540\n---------- Angular Velocity <= -0.02220\n----------- LEFT ENGINE\n----------- LEFT ENGINE\n---------- RIGHT ENGINE\n--------- X Position <= 0.09200\n---------- Angle <= 0.03056\n----------- LEFT ENGINE\n----------- MAIN ENGINE\n---------- LEFT ENGINE\n-------- RIGHT ENGINE\n------ Angle <= 0.02196\n------- X Position <= 0.00000\n-------- RIGHT ENGINE\n-------- NOP\n------- RIGHT ENGINE\n-- Angle <= -0.06016\n--- LEFT ENGINE\n--- Y Velocity <= -0.01140\n---- MAIN ENGINE\n---- NOP"
    config = configs.get_config("lunar_lander")
    # tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
    # config = configs.get_config("mountain_car")
    tree = Individual.read_from_string(config, tree_str)

    rl.fill_metrics(config, [tree], 0, 100, n_jobs=8, should_norm_state=True)

    print(tree)
    print(f"Tree has {tree.get_tree_size()} nodes.")
    print(f"Reward: {tree.reward} +- {tree.std_reward}.")

    tree = reward_pruning(tree, tree, config, episodes=100, should_norm_state=True)

    rl.fill_metrics(config, [tree], 0, 100, n_jobs=8, should_norm_state=True)

    print("---------------------")
    print(tree)
    print(f"Tree has {tree.get_tree_size()} nodes.")
    print(f"Reward: {tree.reward} +- {tree.std_reward}.")