import pdb
import gym
import numpy as np
from erltrees.evo.evo_tree import Individual
from erltrees.io import get_trees_from_logfile
import erltrees.rl.configs as configs

from rich import print
from rich.console import Console

console = Console()

def init_qlearning(tree, config):
    leaves = tree.get_node_list(get_inners=False, get_leaves=True)
    for leaf in leaves:
        # leaf.q_values = np.random.uniform(-1, 1, (config["n_actions"], 1))
        leaf.q_values = np.zeros(config["n_actions"])
        leaf.elig_trace = 0

def run_qlearning(tree, config, episodes=10, lr=0.1, discount=0.9, should_train=True, should_act_by_labels=False, verbose=False):    
    env = config['maker']()
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        lr *= 0.9999

        while not done:
            leaf = tree.get_leaf(state)

            if should_act_by_labels:
                action = tree.act(state)
            else:
                action = np.argmax(leaf.q_values)

            if should_train and np.random.uniform() < 0.5 * (1 - episode / episodes):
                action = np.random.randint(config["n_actions"])
            
            next_state, reward, done, _ = env.step(action)

            if should_train:
                next_leaf = tree.get_leaf(next_state)

                if not done:
                    delta_q = lr * (reward + discount * np.max(next_leaf.q_values) - leaf.q_values[action])
                else:
                    delta_q = lr * (reward + discount * 0 - leaf.q_values[action])
            
                leaf.q_values[action] += delta_q

            state = next_state
            total_reward += reward

        if episode % 10 == 0 and verbose:
            print(f"Episode #{episode} finished with total reward {total_reward}")
            print(lr)
        total_rewards.append(total_reward)
    
    env.close()
    if verbose:
        print(f"Average reward: {np.mean(total_rewards)} +- {np.std(total_rewards)}")
    
    return np.mean(total_rewards), np.std(total_rewards)

def run_watkinsQ(tree, config, episodes=10, lamb=0.9, lr=0.1, discount=0.9, should_train=True, should_act_by_labels=False, verbose=False):    
    env = config['maker']()
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        leaf = tree.get_leaf(state)
        action = np.argmax(leaf.q_values)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_leaf = tree.get_leaf(next_state)

            next_action = np.argmax(next_leaf.q_values)
            if np.random.uniform() < 0.2 * (1 - episode / (episodes * 0.8)):
                next_action = np.random.randint(config["n_actions"])
            
            best_next_action = np.argmax(next_leaf.q_values)

            if not done:
                delta_q = reward + discount * next_leaf.q_values[best_next_action] - leaf.q_values[action]
            else:
                delta_q = reward + discount * 0 - leaf.q_values[action]

            leaf.elig_trace += 1

            for lf in tree.get_node_list(get_inners=False, get_leaves=True):
                lf.q_values[action] += lr * delta_q * lf.elig_trace
                lf.elig_trace *= lamb * discount

            leaf = next_leaf
            action = next_action
            total_reward += reward

        if episode % 10 == 0 and verbose:
            print(f"Episode #{episode} finished with total reward {total_reward}")
            print(lr)
        total_rewards.append(total_reward)
    
    env.close()
    if verbose:
        print(f"Average reward: {np.mean(total_rewards)} +- {np.std(total_rewards)}")
    
    return np.mean(total_rewards), np.std(total_rewards)

def experiment_1():
    # tree_str = "\n- Pole Angular Velocity <= -0.30907\n-- LEFT\n-- Pole Angle <= -0.03442\n--- LEFT\n--- RIGHT"
    tree_str = "\n- Cart Velocity <= 0.39886\n-- Pole Angle <= -0.05032\n--- LEFT\n--- RIGHT\n-- LEFT"
    # tree_str = "\n- Pole Angular Velocity <= -0.10336\n-- Cart Velocity <= -0.60797\n--- RIGHT\n--- Pole Angular Velocity <= -0.48423\n---- LEFT\n---- Pole Angle <= -0.03290\n----- Pole Angular Velocity <= -0.19605\n------ LEFT\n------ RIGHT\n----- LEFT\n-- Pole Angle <= -0.08168\n--- Cart Velocity <= -1.32880\n---- LEFT\n---- LEFT\n--- RIGHT"
    config = configs.get_config("cartpole")
    # tree_str = "\n- Car Velocity <= 0.38800\n-- Car Velocity <= -0.01255\n--- Car Position <= -0.72890\n---- RIGHT\n---- LEFT\n--- Car Position <= -0.10674\n---- RIGHT\n---- LEFT\n-- RIGHT"
    # config = configs.get_config("mountain_car")

    tree = Individual.read_from_string(config, tree_str)
    tree.denormalize_thresholds()
    init_qlearning(tree, config)
    
    run_qlearning(tree, config, episodes=10000, lr=0.2, discount=1, verbose=True, should_train=True)
    print(tree)

    avg_reward, std_reward = run_qlearning(tree, config, episodes=100, verbose=False, should_train=False)
    print(f"Final score: {avg_reward} +- {std_reward}")

def experiment_2():
    filename = "data/complete/cartpole_EVOONLY.txt"
    # filename = "data/cartpole_QHARD.txt"
    config = configs.get_config("cartpole")
    # filename = "data/complete/mountaincar_IL_G.txt"
    # config = configs.get_config("mountain_car")

    print(f"Reading file '{filename}'...")
    tree_strings = get_trees_from_logfile(filename)
    
    history = []

    for i, tree_string in enumerate(tree_strings):
        tree = Individual.read_from_string(config, tree_string)
        tree.denormalize_thresholds()
        
        # Collect traditional reward
        avg_reward, std_reward = run_qlearning(tree, config, episodes=10,
            should_act_by_labels=True, verbose=False, should_train=False)
        
        best_q_avg_reward, best_q_std_reward = 0, 0
        best_reward_difference = 500
        for _ in range(1):
            # Q-Learn
            init_qlearning(tree, config)
            run_qlearning(tree, config, episodes=10, lr=0.2, discount=0.9, verbose=False, should_train=True)

            # Evaluate Q learned behavior
            q_avg_reward, q_std_reward = run_qlearning(tree, config, episodes=20,
                verbose=False, should_train=False)
            # print(f"[yellow]==> Q-learned reward: {q_avg_reward} +- {q_std_reward}[/yellow]")

            reward_difference = abs(avg_reward - q_avg_reward)
            if reward_difference < best_reward_difference:
                best_q_avg_reward = q_avg_reward
                best_q_std_reward = q_std_reward
                best_reward_difference = reward_difference

        print(f"[red]For tree #{i} ({tree.get_tree_size()} nodes):[/red]")
        print(f"[yellow]=> Evolved reward: {avg_reward} +- {std_reward}[/yellow]")
        print(f"[yellow]=> Q-learned reward: {best_q_avg_reward} +- {best_q_std_reward}[/yellow]")
        print(f"[green]=> Difference: {best_reward_difference}[/green]")
        print("--------------")

        history.append((i, tree, reward_difference))

    return history

def experiment_3():
    # tree_str = "\n- Pole Angular Velocity <= -0.30907\n-- LEFT\n-- Pole Angle <= -0.03442\n--- LEFT\n--- RIGHT"
    tree_str = "\n- Cart Velocity <= 0.39886\n-- Pole Angle <= -0.05032\n--- LEFT\n--- RIGHT\n-- LEFT"
    # tree_str = "\n- Pole Angular Velocity <= -0.10336\n-- Cart Velocity <= -0.60797\n--- RIGHT\n--- Pole Angular Velocity <= -0.48423\n---- LEFT\n---- Pole Angle <= -0.03290\n----- Pole Angular Velocity <= -0.19605\n------ LEFT\n------ RIGHT\n----- LEFT\n-- Pole Angle <= -0.08168\n--- Cart Velocity <= -1.32880\n---- LEFT\n---- LEFT\n--- RIGHT"
    config = configs.get_config("cartpole")
    # tree_str = "\n- Car Velocity <= 0.38800\n-- Car Velocity <= -0.01255\n--- Car Position <= -0.72890\n---- RIGHT\n---- LEFT\n--- Car Position <= -0.10674\n---- RIGHT\n---- LEFT\n-- RIGHT"
    # config = configs.get_config("mountain_car")

    tree = Individual.read_from_string(config, tree_str)
    tree.denormalize_thresholds()
    init_qlearning(tree, config)
    
    # run_watkinsQ(tree, config, episodes=1000, lamb=0.9, lr=0.2, discount=0.9, verbose=True, should_train=True)
    run_qlearning(tree, config, episodes=1000, lr=0.2, discount=0.9, verbose=True, should_train=True)
    print(tree)

    avg_reward, std_reward = run_qlearning(tree, config, episodes=100, verbose=False, should_train=False)
    print(f"Final score: {avg_reward} +- {std_reward}")

if __name__ == "__main__":
    experiment_2()