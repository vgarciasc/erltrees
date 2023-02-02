import gym

import pdb
import numpy as np
from rich import print
from sklearn import metrics

import erltrees.rl.utils as rl
import erltrees.rl.configs as configs
import erltrees.io as io

class TreeNode:
    def __init__(self, config, attribute, threshold, 
        label, left=None, right=None, parent=None):

        self.config = config

        self.attribute = attribute
        self.threshold = threshold
        self.label = label

        self.left = left
        self.right = right
        self.parent = parent

        self.visits = 0

    def __str__(self):
        return f"[attrib: {self.attribute}, threshold: {self.threshold}, " + \
            f"label: {self.label}, is_leaf: {self.is_leaf()}]"
    
    def copy(self):
        if self.is_leaf:
            return TreeNode(self.config, self.attribute, self.threshold, self.label)
        else:
            new_left = self.left.copy()
            new_right = self.right.copy()

            new_node = TreeNode(self.config, self.attribute, 
                self.threshold, self.label, new_left, new_right)
            
            new_left.parent = new_node
            new_right.parent = new_node

            return new_node
    
    def is_root(self):
        return self.parent is None
    
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def is_left_child(self):
        if self.is_root():
            return False
        return self.parent.left == self

    def is_right_child(self):
        if self.is_root():
            return False
        return self.parent.right == self
    
    def get_path(self, path=[]):
        if self.is_root():
            return path

        if self.is_left_child():
            return self.parent.get_path(path) + ["left"] 
        else:
            return self.parent.get_path(path) + ["right"] 

    def get_node_by_path(self, path):
        node = self
        for direction in path:
            if hasattr(node, direction):
                node = getattr(node, direction)
            else:
                return None
        return node

    def get_leaf(self, state):
        self.visits += 1

        if self.is_leaf():
            return self
        
        if state[self.attribute] <= self.threshold:
            return self.left.get_leaf(state)
        else:
            return self.right.get_leaf(state)

    def act(self, state):
        return self.get_leaf(state).label
    
    def get_tree_size(self):
        total = 1
        if self.left != None:
            total += self.left.get_tree_size() 
        if self.right != None:
            total += self.right.get_tree_size() 
        return total
    
    def get_height(self):
        if self.is_leaf():
            return 1
        
        left_height = self.left.get_height()
        right_height = self.right.get_height()
        
        return max(left_height, right_height) + 1
    
    def replace_node(self, node_src, node_dst):
        if node_src.parent != None:
            if node_src.parent.left == node_src:
                node_src.parent.left = node_dst
            else:
                node_src.parent.right = node_dst
    
    def __str__(self, include_visits=False):
        stack = [(self, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if node.is_leaf():
                output += (self.config['actions'][node.label]).upper()
                if include_visits:
                    output += f" (visits: {node.visits})"
                # output += (self.config['actions'][np.argmax(node.q_values)]).upper() + " " + str(node.q_values)
                if hasattr(node, "q_values"):
                    output += ", ".join([str((action_name, node.q_values[action_id])) + ("*" if np.argmax(node.q_values) == action_id else "") for action_id, action_name in enumerate(self.config["actions"])])
            else:
                output += self.config['attributes'][node.attribute][0]
                output += " <= "
                output += '{:.5f}'.format(node.threshold)
                
                if node.right:
                    stack.append((node.right, depth + 1))
                if node.left:
                    stack.append((node.left, depth + 1))
            output += "\n"

        return output
    
    def is_equal(node_a, node_b):
        if not node_a.is_leaf() and not node_b.is_leaf():
            return node_a.threshold == node_b.threshold and \
                node_a.attribute == node_b.attribute

        if node_a.is_leaf() and node_b.is_leaf():
            return node_a.label == node_b.label
    
    def distance(self, tree):
        node_list = tree.get_node_list()
        stack = [self]
        output = 0

        while len(stack) > 0:
            node = stack.pop()

            if not node.is_leaf():
                counterparts = [n for n in node_list if TreeNode.is_equal(n, node)]
                output += 1 if len(counterparts) > 0 else 0
                
                if node.right:
                    stack.append((node.right))
                if node.left:
                    stack.append((node.left))

        if output == 0:
            return 0
        
        max_inner_nodes = (max([len(node_list), self.get_tree_size()]) / 2 - 0.5)
        return output / max_inner_nodes
    
    def get_node_list(self, get_inners=True, get_leaves=True):
        stack = [self]
        output = []

        while len(stack) > 0:
            node = stack.pop()
            if (node.is_leaf() and get_leaves) or (not node.is_leaf() and get_inners):
                output.append(node)

            if not node.is_leaf():
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)

        return output
    
    def get_leaf_mask(self, mask=[], path=[]):
        if self.is_leaf():
            mask.append(path)
            return mask
        
        l_path = path + [-1 * self.right.get_tree_size()]
        l_mask = self.left.get_leaf_mask(mask, l_path)

        r_path = path + [+1 * self.left.get_tree_size()] + [0] * (len(mask[-1]) - len(path) - 1)
        r_mask = self.right.get_leaf_mask(l_mask, r_path)

        if self.is_root():
            max_len = np.max([len(m_i) for m_i in r_mask])
            r_mask = [m_i + [0] * (max_len - len(m_i)) for m_i in r_mask]
            r_mask = np.array(r_mask)
        
        return r_mask
    
    def get_weight_matrix(self, W = []):
        if self.is_leaf():
            return W
        
        weights = [0] * (self.config["n_attributes"] + 1)
        weights[0] = -self.threshold
        weights[self.attribute + 1] = 1

        W += [weights]
        W = self.left.get_weight_matrix(W)
        W = self.right.get_weight_matrix(W)
    
        if self.is_root():
            W = np.array(W)
        
        return W
    
    def get_label_vector(self, L=[]):
        if self.is_leaf():
            return L + [self.label]
        
        L = self.left.get_label_vector(L)
        L = self.right.get_label_vector(L)

        if self.is_root():
            L = np.array(L)

        return L

    def act_by_matrix(self, state, W, labels, mask):
        WxT = W[:,1:] @ state + W[:,0]
        K = mask @ np.sign(WxT)
        leaf = np.argmax(K)

        return labels[leaf]

    def act_by_matrix_batch(self, X, W, labels, mask):
        WxT = W @ np.c_[np.ones(len(X)), X].T
        K = mask @ np.sign(WxT)
        # L = np.clip(K - (np.max(K) - 1), 0, 1)
        leaves = np.argmax(K.T, axis=1)
        y = np.array([labels[l] for l in leaves])

        return y
    
    def normalize_thresholds(self):
        (_, _, (xmin, xmax)) = self.config["attributes"][self.attribute]
        if abs(xmax) > 9999:
            xmax = 1
        if abs(xmin) > 9999:
            xmin = -1

        self.threshold = (self.threshold - xmin) / (xmax - xmin) * 2 - 1

        if not self.is_leaf():
            self.left.normalize_thresholds()
            self.right.normalize_thresholds()
    
    def denormalize_thresholds(self):
        (_, _, (xmin, xmax)) = self.config["attributes"][self.attribute]
        if abs(xmax) > 9999:
            xmax = 1
        if abs(xmin) > 9999:
            xmin = -1
        
        self.threshold = (self.threshold + 1) * (xmax - xmin) / 2 + xmin

        if self.left is not None:
            self.left.denormalize_thresholds()
        if self.right is not None:
            self.right.denormalize_thresholds()

if __name__ == "__main__":
    config = configs.get_config("cartpole")
    tree = TreeNode(config, 2, -0.037, 1, False, 
        left=TreeNode(config, 2, 0.01, 0, True),
        right=TreeNode(config, 3, -0.689, 0, False, 
            left=TreeNode(config, 1, 2.1, 0, True),
            right=TreeNode(config, 1, 0.2, 1, True)))
    
    # config = get_config("mountain_car")
    # tree = TreeNode(config, 0, 0.158, 1, False, 
    #     left=TreeNode(config, 1, 0.000, 1, False,
    #         left=TreeNode(config, 1, 2.1, 0, True),
    #         right=TreeNode(config, 1, 0.2, 2, True)),
    #     right=TreeNode(config, 2, -0.41, 2, True, 
    #         left=TreeNode(config, 1, 2.1, 0, True),
    #         right=TreeNode(config, 1, 0.2, 1, True)))

    io.printv(tree, verbose=True)
    metrics = rl.collect_metrics(config, [tree], episodes=100,
                                 should_norm_state=True,
                                 render=True)

    print("[yellow]> Evaluating fitness:[/yellow]")
    print(f"Mean reward, std reward: {metrics}")