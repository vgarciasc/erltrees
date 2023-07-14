import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF

from erltrees.il.classification_tree import ClassificationTree


class RegressorTree(ClassificationTree):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y, pruning=0):
        clf = tree.DecisionTreeRegressor(ccp_alpha=pruning)
        clf = clf.fit(X, y)
        self.model = clf

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
                content = values[node_id][0][0]
                if show_prob:
                    prob = np.max(values[node_id][0]) / sum(values[node_id][0])
                    content += f" ({'{:.2f}'.format(prob)})"
            else:
                content = self.config['attributes'][feature[node_id]][0] + " <= " + str(threshold[node_id])

                stack.append((children_right[node_id], depth + 1))
                stack.append((children_left[node_id], depth + 1))

            output += f"\n{'-' * depth} {content}"

        return output