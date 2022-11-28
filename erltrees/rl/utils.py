from gc import collect
import pdb
from time import time
import numpy as np
import gym

from joblib import Parallel, delayed

norm_state_mins = None
norm_state_maxs = None

### I/O functions

def normalize_state(config, state):
    global norm_state_mins
    global norm_state_maxs

    if norm_state_mins is None:
        norm_state_mins = np.array([(xmin if abs(xmin) < 9999 else -1) for (_, _, (xmin, xmax)) in config["attributes"]])
    if norm_state_maxs is None:
        norm_state_maxs = np.array([(xmax if abs(xmax) < 9999 else +1) for (_, _, (xmin, xmax)) in config["attributes"]])
    
    return (state - norm_state_mins) / (norm_state_maxs - norm_state_mins) * 2 - 1

def calc_fitness_tree(tree, alpha, should_penalize_std):
    penalized_reward = (tree.reward - tree.std_reward) if should_penalize_std else tree.reward
    return penalized_reward - alpha * tree.get_tree_size()

def calc_fitness(mean_reward, std_reward, tree_size, alpha, should_penalize_std):
    penalized_reward = (mean_reward - std_reward) if should_penalize_std else mean_reward
    return penalized_reward - alpha * tree_size

def run_episode(tree, env, config, should_norm_state=False, render=False):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if should_norm_state:
            state = normalize_state(config, state)
        
        action = tree.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()
    
    return total_reward

def collect_rewards_seq(config, tree, episodes, should_norm_state, render=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        total_reward = run_episode(tree, env, config, should_norm_state, render)
        total_rewards.append(total_reward)
    
    env.close()
    return total_rewards

def collect_rewards_par(config, tree, episodes, should_norm_state, n_jobs=4):
    episodes_partition = [episodes // n_jobs for _ in range(n_jobs)]
    episodes_partition[-1] += episodes % n_jobs

    total_rewards = Parallel(n_jobs=n_jobs)(delayed(collect_rewards_seq)(
        config, tree, partition, should_norm_state=should_norm_state)
        for partition in episodes_partition)
    total_rewards = [item for sublist in total_rewards for item in sublist]

    return total_rewards

def collect_rewards(config, tree, episodes, should_norm_state, render=False, n_jobs=-1):
    if n_jobs == -1:
        return collect_rewards_seq(config, tree, episodes, 
            should_norm_state=should_norm_state, render=render)
    else:
        return collect_rewards_par(config, tree, episodes, 
            should_norm_state=should_norm_state, n_jobs=n_jobs)

def calc_metrics(tree, rewards, alpha, penalize_std, task_solution_threshold):
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if hasattr(tree, "get_tree_size"):
        fitness = calc_fitness(
            avg_reward, std_reward, 
            tree.get_tree_size(), alpha, penalize_std)
    else:
        fitness = None

    if task_solution_threshold is not None:
        success_rate = np.mean([(1 if r > task_solution_threshold else 0) for r in rewards])
    else:
        success_rate = None

    return avg_reward, std_reward, fitness, success_rate

def collect_metrics(config, trees, alpha=0.5, episodes=10,
    should_norm_state=False, penalize_std=False,
    should_fill_attributes=False, task_solution_threshold=None,
    render=False, verbose=False, n_jobs=-1):

    output = []

    env = gym.make(config["name"])

    for tree in trees:
        rewards = collect_rewards(config, tree, episodes, 
            should_norm_state=should_norm_state, render=render, 
            n_jobs=n_jobs)

        metrics = calc_metrics(tree, rewards, alpha, 
            penalize_std, task_solution_threshold)
        avg_reward, std_reward, fitness, success_rate = metrics

        if should_fill_attributes:
            tree.reward = avg_reward
            tree.std_reward = std_reward
            tree.fitness = fitness
            tree.success_rate = success_rate

        output.append((avg_reward, std_reward, fitness))

    env.close()

    return output

# def collect_metrics(config, trees, alpha=0.5, episodes=10,
#     should_norm_state=False, penalize_std=False,
#     should_fill_attributes=False,
#     render=False, verbose=False):

#     output = []

#     env = gym.make(config["name"])

#     for tree in trees:
#         total_rewards = []

#         W = tree.get_weight_matrix(W=[])
#         labels = tree.get_label_vector(L=[])
#         mask = tree.get_leaf_mask(mask=[], path=[])

#         for episode in range(episodes):
#             state = env.reset()
#             total_reward = 0
#             done = False

#             while not done:
#                 if should_norm_state:
#                     state = normalize_state(config, state)
                
#                 # action = tree.act(state)
#                 action = tree.act_by_matrix(state, W, labels, mask)
                
#                 state, reward, done, _ = env.step(action)
#                 total_reward += reward

#                 if render:
#                     env.render()
            
#             total_rewards.append(total_reward)

#         tree_avg_reward = np.mean(total_rewards)
#         tree_std_reward = np.std(total_rewards)
#         tree_fitness = calc_fitness(
#             tree_avg_reward, tree_std_reward, 
#             tree.get_tree_size(), alpha, penalize_std)

#         if should_fill_attributes:
#             tree.reward = tree_avg_reward
#             tree.std_reward = tree_std_reward
#             tree.fitness = tree_fitness

#         output.append((tree_avg_reward, tree_std_reward, tree_fitness))

#     env.close()

#     return output

# def collect_metrics(config, trees, alpha=0.5, episodes=10,
#     should_norm_state=False, penalize_std=False,
#     should_fill_attributes=False,
#     render=False, verbose=False):

#     output = []

#     envs = [gym.make(config["name"]) for _ in range(episodes)]

#     for tree in trees:
#         if should_norm_state:
#             tree_2 = tree.copy()
#             tree_2.denormalize_thresholds()
#             W = tree_2.get_weight_matrix(W=[])
#         else:
#             W = tree.get_weight_matrix(W=[])

#         labels = tree.get_label_vector(L=[])
#         mask = tree.get_leaf_mask(mask=[], path=[])

#         X = np.array([env.reset() for env in envs])
#         dones = [False for env in envs]
#         done_number = 0
#         total_rewards = np.zeros(episodes)

#         while done_number < episodes:
#             actions = tree.act_by_matrix_batch(X, W, labels, mask)

#             X_next = np.zeros_like(X)
#             for i, env in enumerate(envs):
#                 if dones[i]:
#                     # candidate for refactor
#                     continue

#                 # if should_norm_state:
#                 #     state = normalize_state(config, X[i])
#                 # action = tree.act(state)
#                 state, reward, done, _ = env.step(actions[i])
#                 X_next[i] = state
#                 total_rewards[i] += reward

#                 if done:
#                     dones[i] = True
#                     done_number += 1
                
#             X = X_next

#         tree_avg_reward = np.mean(total_rewards)
#         tree_std_reward = np.std(total_rewards)
#         tree_fitness = calc_fitness(
#             tree_avg_reward, tree_std_reward, 
#             tree.get_tree_size(), alpha, penalize_std)

#         if should_fill_attributes:
#             tree.reward = tree_avg_reward
#             tree.std_reward = tree_std_reward
#             tree.fitness = tree_fitness

#         output.append((tree_avg_reward, tree_std_reward, tree_fitness))
        
#     for env in envs:
#         env.close()

#     return output

def fill_metrics_par(n_jobs, config, trees, alpha, episodes=10, 
    should_norm_state=False, task_solution_threshold=None, penalize_std=False):

    partitions = [trees[len(trees)//n_jobs * i : len(trees)//n_jobs * (i+1)] 
        for i in range(n_jobs)]
    partitions[-1] = trees[len(trees)//n_jobs * (n_jobs-1) :]
    
    metrics = Parallel(n_jobs=n_jobs)(delayed(collect_metrics)(config, 
        partition, alpha=alpha, episodes=episodes, 
        should_norm_state=should_norm_state, 
        task_solution_threshold=task_solution_threshold,
        penalize_std=penalize_std) for partition in partitions)
    metrics = [tree_metric for tree_metrics in metrics for tree_metric in tree_metrics]

    for (i, (reward, std_reward, fitness)) in enumerate(metrics):
        trees[i].reward = reward
        trees[i].std_reward = std_reward
        trees[i].fitness = fitness

def fill_metrics_seq(config, trees, alpha, episodes=10,
    should_norm_state=False, task_solution_threshold=None, penalize_std=False):

    collect_metrics(config, trees, alpha, episodes=episodes,
        should_norm_state=should_norm_state,
        penalize_std=penalize_std, should_fill_attributes=True,
        task_solution_threshold=task_solution_threshold,
        render=False, verbose=False)

def fill_metrics(config, trees, alpha, episodes=10, 
    should_norm_state=False, penalize_std=False,
    task_solution_threshold=None, n_jobs=-1):

    if n_jobs == -1:
        fill_metrics_seq(config, trees, alpha, episodes=episodes,
            should_norm_state=should_norm_state,
            task_solution_threshold=task_solution_threshold,
            penalize_std=penalize_std)
    else:
        fill_metrics_par(n_jobs, 
            config, trees, alpha, episodes=episodes, 
            should_norm_state=should_norm_state,
            task_solution_threshold=task_solution_threshold,
            penalize_std=penalize_std)

def collect_and_prune_by_visits(tree, threshold=5, episodes=100, 
    should_norm_state=True):

    collect_metrics(tree.config, [tree], episodes=episodes,
        should_norm_state=should_norm_state)
    tree.prune_by_visits(threshold)