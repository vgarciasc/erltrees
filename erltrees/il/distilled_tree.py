import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF

class DistilledTree:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y, pruning=0):
        clf = tree.DecisionTreeClassifier(ccp_alpha=pruning)
        clf = clf.fit(X, y)
        self.model = clf

    def act(self, state):
        state = state.reshape(1, -1)
        action = self.model.predict(state)
        action = action[0]
        return action
    
    def save_fig(self):
        plt.figure(figsize=(25, 25))
        feature_names = [name for (name, _, _, _) in self.config["attributes"]]
        tree.plot_tree(self.model, feature_names=feature_names)
        plt.savefig('last_tree.png')

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved tree to '{filename}'.")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        
    def get_size(self):
        return self.model.get_n_leaves() * 2 - 1

    def get_as_viztree(self, show_prob=False):
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        values = self.model.tree_.value

        stack = [(0, 1)]
        output = ""

        while len(stack) > 0:
            node_id, depth = stack.pop()
            is_leaf = children_left[node_id] == children_right[node_id]

            if is_leaf:
                content = self.config['actions'][self.model.classes_[np.argmax(values[node_id][0])]].upper()
                if show_prob:
                    prob = np.max(values[node_id][0]) / sum(values[node_id][0])
                    content += f" ({'{:.2f}'.format(prob)})"
            else:
                content = self.config['attributes'][feature[node_id]][0] + " <= " + str(threshold[node_id])

                stack.append((children_right[node_id], depth + 1))
                stack.append((children_left[node_id], depth + 1))

            output += f"\n{'-' * depth} {content}"

        return output
    
    def prune_redundant_leaves(self):
        DistilledTree.prune_duplicate_leaves(self.model)

    def is_leaf(inner_tree, index):
        # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and 
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_index(inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not DistilledTree.is_leaf(inner_tree, inner_tree.children_left[index]):
            DistilledTree.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not DistilledTree.is_leaf(inner_tree, inner_tree.children_right[index]):
            DistilledTree.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:     
        if (DistilledTree.is_leaf(inner_tree, inner_tree.children_left[index]) and
            DistilledTree.is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and 
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            ##print("Pruned {}".format(index))

    def prune_duplicate_leaves(mdl):
        # Remove leaves if both 
        decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
        DistilledTree.prune_index(mdl.tree_, decisions)