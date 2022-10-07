from gc import collect
import pdb
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

def collect_metrics(config, trees, alpha=0.5, episodes=10,
    should_norm_state=False, penalize_std=False,
    should_fill_attributes=False,
    render=False, verbose=False):

    output = []

    for tree in trees:
        env = gym.make(config["name"])
        total_rewards = []

        for episode in range(episodes):
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

            # printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
            total_rewards.append(total_reward)
        
        env.close()

        tree_avg_reward = np.mean(total_rewards)
        # tree_avg_reward = np.min(total_rewards)
        tree_std_reward = np.std(total_rewards)
        tree_fitness = calc_fitness(
            tree_avg_reward, tree_std_reward, 
            tree.get_tree_size(), alpha, penalize_std)

        if should_fill_attributes:
            tree.reward = tree_avg_reward
            tree.std_reward = tree_std_reward
            tree.fitness = tree_fitness

        output.append((tree_avg_reward, tree_std_reward, tree_fitness))

    return output

def fill_metrics_par(n_jobs, config, trees, alpha, episodes=10, 
    should_norm_state=False, penalize_std=False):

    partitions = [trees[len(trees)//n_jobs * i : len(trees)//n_jobs * (i+1)] 
        for i in range(n_jobs)]
    partitions[-1] = trees[len(trees)//n_jobs * (n_jobs-1) :]
    
    metrics = Parallel(n_jobs=n_jobs)(delayed(collect_metrics)(config, 
        partition, alpha=alpha, episodes=episodes, 
        should_norm_state=should_norm_state, 
        penalize_std=penalize_std) for partition in partitions)
    metrics = [tree_metric for tree_metrics in metrics for tree_metric in tree_metrics]

    for (i, (reward, std_reward, fitness)) in enumerate(metrics):
        trees[i].reward = reward
        trees[i].std_reward = std_reward
        trees[i].fitness = fitness

def fill_metrics_seq(config, trees, alpha, episodes=10,
    should_norm_state=False, penalize_std=False):

    collect_metrics(config, trees, alpha, episodes=episodes,
        should_norm_state=should_norm_state,
        penalize_std=penalize_std, should_fill_attributes=True,
        render=False, verbose=False)

def fill_metrics(config, trees, alpha, episodes=10, 
    should_norm_state=False, penalize_std=False, 
    n_jobs=-1):

    if n_jobs == -1:
        fill_metrics_seq(config, trees, alpha, episodes=episodes,
            should_norm_state=should_norm_state,
            penalize_std=penalize_std)
    else:
        fill_metrics_par(n_jobs, 
            config, trees, alpha, episodes=episodes, 
            should_norm_state=should_norm_state,
            penalize_std=penalize_std)

def collect_and_prune_by_visits(tree, threshold=5, episodes=100, 
    should_norm_state=True):

    collect_metrics(tree.config, [tree], episodes=episodes,
        should_norm_state=should_norm_state)
    tree.prune_by_visits(threshold)