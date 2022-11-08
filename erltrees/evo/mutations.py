import pdb
import numpy as np
import erltrees.io as io
import erltrees.rl.utils as rl

def mutate(tree, mutation="A"):
    if mutation == "A":
        mutate_A(tree)
    elif mutation == "B":
        mutate_B(tree)
    elif mutation == "C":
        mutate_C(tree)
    elif mutation == "D":
        mutate_D(tree)
    elif mutation == "E":
        mutate_E(tree)
    elif mutation == "F":
        mutate_F(tree)
    elif mutation == "G":
        mutate_G(tree)
    elif mutation == "G2":
        mutate_G2(tree)
    elif mutation == "H":
        mutate_H(tree)
    elif mutation == "I":
        mutate_I(tree)
    elif mutation == "I2":
        mutate_I2(tree)
    elif mutation == "I3":
        mutate_I3(tree)
    else:
        raise(f"Mutation {mutation} not found.")

def mutate_A(tree, top_splits=[]):
    node = tree.get_random_node()

    if node.is_leaf():
        operation = np.random.choice(["label", "is_leaf"])
        
        if operation == "label":
            node.mutate_label()
        elif operation == "is_leaf":
            node.mutate_is_leaf()
    else:
        operation = np.random.choice(["attribute", "threshold", "is_leaf"])
        
        if operation == "attribute":
            if top_splits != [] and np.random.uniform(0, 1) <= 0.5:
                attribute, threshold = top_splits[np.random.randint(0, len(top_splits))]
                # print(f"Reusing top split ({attribute}, {threshold}).")
                node.attribute = attribute
                node.threshold = threshold
            else:
                node.mutate_attribute()
                node.threshold = np.random.uniform(-1, 1)
        elif operation == "threshold":
            node.mutate_threshold()
        elif operation == "is_leaf":
            node.mutate_is_leaf()

def mutate_B(tree, sigma=None):
    operation = np.random.choice(
        ["leaf_label", "inner_attribute",
            "inner_threshold", "is_leaf", 
            "cut_parent"])

    if operation == "leaf_label":
        leaf = tree.get_random_node(get_inners=False, get_leaves=True)
        leaf.mutate_label()
        return 0
    elif operation == "inner_attribute" or operation == "inner_threshold":
        inner = tree.get_random_node(get_inners=True, get_leaves=False)
        
        if operation == "inner_attribute":
            inner.mutate_attribute(force_change=False)
            inner.threshold = np.random.uniform(-1, 1)
            return 1
        elif operation == "inner_threshold":
            inner.threshold += np.random.normal(0, 1)
            return 2
    elif operation == "is_leaf":
        node = tree.get_random_node()
        node.mutate_is_leaf()
        return 3
    elif operation == "cut_parent":
        node = tree.get_random_node()
        node.cut_parent()
        return 4

def mutate_C(tree, sigma=None):
    node_list = tree.get_node_list()
    probabilities = [1 / node.get_height() for node in node_list]
    probabilities /= np.sum(probabilities)
    node = np.random.choice(node_list, p=probabilities)

    if node.is_leaf():
        operation = np.random.choice(["label", "is_leaf", "cut_parent"])
        
        if operation == "label":
            node.mutate_label()
        elif operation == "is_leaf":
            node.mutate_is_leaf()
        elif operation == "cut_parent":
            node.cut_parent()
    else:
        operation = np.random.choice(["attribute", "threshold", "is_leaf", "cut_parent"])
        
        if operation == "attribute":
            node.mutate_attribute()
            node.threshold = np.random.uniform(-1, 1)
        elif operation == "threshold":
            node.mutate_threshold()
        elif operation == "is_leaf":
            node.mutate_is_leaf()
        elif operation == "cut_parent":
            node.cut_parent()

def mutate_D(tree, top_splits=[], verbose=False):
    operation = np.random.choice(["add", "remove", "modify"], p=[0.45, 0.1, 0.45])

    if operation == "add":
        node = tree.get_random_node()

        if node.is_leaf():
            node.mutate_is_leaf()
        else:
            node.mutate_truncate()

    elif operation == "remove":
        node_list = tree.get_node_list()
        probabilities = [1 / node.get_height() for node in node_list]
        probabilities /= np.sum(probabilities)
        node = np.random.choice(node_list, p=probabilities)

        if node.is_leaf():
            node.cut_parent()
        else:
            node.replace_child()

    elif operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label()
        else:
            operation = np.random.choice(["attribute", "threshold"])
            
            if operation == "attribute":
                if top_splits != [] and np.random.uniform(0, 1) <= 0.5:
                    attribute, threshold = top_splits[np.random.randint(0, len(top_splits))]
                    io.printv(f"Reusing top split ({attribute}, {threshold}).", verbose)
                    node.attribute = attribute
                    node.threshold = threshold
                else:
                    node.mutate_attribute()
                    node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold()

def mutate_E(tree, top_splits=[], verbose=False):
    operation = np.random.choice(["add", "remove", "modify"], p=[0.4, 0.2, 0.4])

    if operation == "add":
        node = tree.get_random_node()

        if node.is_leaf():
            node.mutate_is_leaf()
        else:
            node.mutate_add_inner_node()

    elif operation == "remove":
        node_list = tree.get_node_list()
        probabilities = [1 / node.get_height() for node in node_list]
        probabilities /= np.sum(probabilities)
        node = np.random.choice(node_list, p=probabilities)

        if node.is_leaf():
            node.cut_parent()
        else:
            node.replace_child()

    elif operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label()
        else:
            operation = np.random.choice(["attribute", "threshold"])
            
            if operation == "attribute":
                if top_splits != [] and np.random.uniform(0, 1) <= 0.5:
                    attribute, threshold = top_splits[np.random.randint(0, len(top_splits))]
                    io.printv(f"Reusing top split ({attribute}, {threshold}).", verbose)
                    node.attribute = attribute
                    node.threshold = threshold
                else:
                    node.mutate_attribute()
                    node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold()

def mutate_F(tree, removal_depth_param=2, verbose=False):
    operation = np.random.choice(["add", "remove", "modify"], p=[0.2, 0.2, 0.6])
    
    if operation == "add":
        node = tree.get_random_node()

        if node.is_leaf():
            node.mutate_is_leaf(verbose)
        else:
            node.mutate_add_inner_node(verbose)

    if operation == "remove":
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
    
    if operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"], p=[1/3, 2/3])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(verbose)

def mutate_G(tree, removal_depth_param=1, p_add=0.5, p_remove=0.5, p_modify=0.5, p_prune=0.1, verbose=False):
    if np.random.uniform(0, 1) < p_add:
        # Add
        node = tree.get_random_node()

        if node.is_leaf():
            node.mutate_is_leaf(verbose)
        else:
            node.mutate_add_inner_node(verbose)

    if np.random.uniform(0, 1) < p_remove:
        # Remove
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
        
    if np.random.uniform(0, 1) < p_modify:
        # Modify
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)
        
    if np.random.uniform(0, 1) < p_prune:
        # Prune by visits
        rl.collect_and_prune_by_visits(tree, episodes=20)

def mutate_G2(tree, removal_depth_param=1, p_add=0.5, p_remove=0.5, p_modify=0.5, p_prune=0.1, verbose=False):
    if np.random.uniform(0, 1) < p_add:
        # Add
        node = tree.get_random_node()

        if node.is_leaf():
            node.mutate_is_leaf(verbose)
        else:
            node.mutate_add_inner_node(verbose)

    if np.random.uniform(0, 1) < p_remove:
        # Remove
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        # probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        # probabilities /= np.sum(probabilities)

        # node = np.random.choice(node_list, p=probabilities)
        node = np.random.choice(node_list)
        node.mutate_truncate_dx(verbose)
        
    if np.random.uniform(0, 1) < p_modify:
        # Modify
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)
        
    if np.random.uniform(0, 1) < p_prune:
        rl.collect_and_prune_by_visits(tree, episodes=20)

def mutate_H(tree, removal_depth_param=2, verbose=False):
    operation = np.random.choice(["expand_leaf", "add_inner_node", "remove_risky", "remove_cautious", "modify"], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    if operation == "expand_leaf":
        node = tree.get_random_node(get_inners=False, get_leaves=True)
        node.mutate_is_leaf(verbose)

    if operation == "remove_risky":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_truncate_dx(verbose)

    if operation == "remove_cautious":
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
    
    if operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"], p=[1/3, 2/3])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)
    
    if operation == "prune":
        rl.collect_and_prune_by_visits(tree)

def mutate_I(tree, removal_depth_param=2, verbose=False):
    operation = np.random.choice(["expand_leaf", "add_inner_node", "remove_risky", "remove_cautious", "modify"], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    if operation == "expand_leaf":
        node = tree.get_random_node(get_inners=False, get_leaves=True)
        node.mutate_is_leaf(verbose)

    if operation == "add_inner_node":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_add_inner_node(verbose)

    if operation == "remove_risky":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_truncate_dx(verbose)

    if operation == "remove_cautious":
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
    
    if operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"], p=[1/3, 2/3])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)

def mutate_I2(tree, removal_depth_param=2, verbose=False):
    operation = np.random.choice(["expand_leaf", "add_inner_node", "remove_risky", "remove_cautious", "modify"], p=[0.1, 0.1, 0.2, 0.4, 0.2])
    
    if operation == "expand_leaf":
        node = tree.get_random_node(get_inners=False, get_leaves=True)
        node.mutate_is_leaf(verbose)

    if operation == "add_inner_node":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_add_inner_node(verbose)

    if operation == "remove_risky":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_truncate_dx(verbose)

    if operation == "remove_cautious":
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
    
    if operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"], p=[1/3, 2/3])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)

def mutate_I3(tree, removal_depth_param=2, verbose=False):
    operation = np.random.choice(["expand_leaf", "add_inner_node", "remove_risky", "remove_cautious", "modify"], p=[0.1, 0.1, 0.4, 0.1, 0.3])
    
    if operation == "expand_leaf":
        node = tree.get_random_node(get_inners=False, get_leaves=True)
        node.mutate_is_leaf(verbose)

    if operation == "add_inner_node":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_add_inner_node(verbose)

    if operation == "remove_risky":
        node = tree.get_random_node(get_inners=True, get_leaves=False)
        node.mutate_truncate_dx(verbose)

    if operation == "remove_cautious":
        node_list = tree.get_node_list(get_inners=True, get_leaves=False)
        
        probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
        probabilities /= np.sum(probabilities)

        node = np.random.choice(node_list, p=probabilities)
        node.replace_child(verbose)
    
    if operation == "modify":
        node = tree.get_random_node()
        
        if node.is_leaf():
            node.mutate_label(verbose)
        else:
            operation = np.random.choice(["attribute", "threshold"], p=[1/3, 2/3])
            
            if operation == "attribute":
                node.mutate_attribute(verbose)
                node.threshold = np.random.uniform(-1, 1)
            elif operation == "threshold":
                node.mutate_threshold(0.1, verbose)