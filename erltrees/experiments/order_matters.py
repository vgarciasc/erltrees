import numpy as np
import pdb

CLOCKWISE = 1
COUNTERCLOCKWISE = 0

def step(state, action, reward_vec):
    transition_matrix = [[3, 3], [3, 0], [3, 0], [2, 1]]

    next_state = transition_matrix[state][action]
    reward = reward_vec[state][action]

    return next_state, reward

def evaluate_tree(reward_vec, tree_func, actions, steps=1000):
    state = 0
    total_reward = 0
    
    for _ in range(steps):
        action = tree_func(state, actions)
        state, reward = step(state, action, reward_vec)
        total_reward += reward

    return total_reward / steps

def simple_tree_full(state, actions):
    if state <= 1:
        if state <= 0:
            return actions[0]
        else:
            return actions[1]
    else:
        if state <= 2:
            return actions[2]
        else:
            return actions[3]

def simple_tree_1(state, actions):
    if state <= 1:
        if state <= 0:
            return actions[0]
        else:
            return actions[1]
    else:
        return actions[2]

def simple_tree_2(state, actions):
    if state <= 1:
        return actions[0]
    else:
        if state <= 2:
            return actions[1]
        else:
            return actions[2]

def simplest_tree(state, actions):
    if state <= 1:
        return actions[0]
    else:
        return actions[1]

if __name__ == "__main__":
    # reward_vec = [[1, 3], [10, 12], [21, 23], [32, 30]]
    reward_vec = [[0.2, 0.9], [0.3, 0.6], [0.8, 0.7], [0.5, 0.7]]

    actions = [0, 1, 0, 1]
    max_iter = 1000000

    # for iter in range(max_iter):
        # print(f"\nEvaluating reward_vec:")
        # print(reward_vec)

    actions = [0, 1, 0, 1]
    full_tree_reward = evaluate_tree(reward_vec, simple_tree_full, actions)
    # print(f"Full tree: {full_tree_reward}")
    
    actions = [0, 1, 1]
    simple_tree_1_reward_1 = evaluate_tree(reward_vec, simple_tree_1, actions)
    actions = [0, 1, 0]
    simple_tree_1_reward_2 = evaluate_tree(reward_vec, simple_tree_1, actions)
    simple_tree_1_reward = max(simple_tree_1_reward_1, simple_tree_1_reward_2)
    # print(f"Simple tree 2: {simple_tree_2_reward}")
    
    second_leaf_val = [1, 0][np.argmax([simple_tree_1_reward_1, simple_tree_1_reward_2])]
    actions = [1, second_leaf_val]
    simplest_tree_reward_1 = evaluate_tree(reward_vec, simplest_tree, actions)
    actions = [0, second_leaf_val]
    simplest_tree_reward_2 = evaluate_tree(reward_vec, simplest_tree, actions)
    # print(f"Simplest tree: {simplest_tree_reward}")
    
    actions = [0, 0, 1]
    simple_tree_2_reward_1 = evaluate_tree(reward_vec, simple_tree_2, actions)
    actions = [1, 0, 1]
    simple_tree_2_reward_2 = evaluate_tree(reward_vec, simple_tree_2, actions)
    simple_tree_2_reward = max(simple_tree_2_reward_1, simple_tree_2_reward_2)
    # print(f"Simple tree 1: {simple_tree_1_reward}")

    if simple_tree_1_reward > full_tree_reward and \
        simple_tree_2_reward > full_tree_reward and \
        simple_tree_2_reward > simple_tree_1_reward and \
        simple_tree_2_reward > simplest_tree_reward_1 and \
        simple_tree_2_reward > simplest_tree_reward_2:
        
        print(f"\nEvaluating reward_vec:")
        print(reward_vec)
        print(f"Full tree: {full_tree_reward}")
        print(f"Simple tree 1: {simple_tree_1_reward} ({simple_tree_1_reward_1}, {simple_tree_1_reward_2})")
        print(f"Simple tree 2: {simple_tree_2_reward} ({simple_tree_2_reward_1}, {simple_tree_2_reward_2})")
        print(f"Simplest tree 1: {simplest_tree_reward_1}")
        print(f"Simplest tree 2: {simplest_tree_reward_2}")
        print("Found!")
