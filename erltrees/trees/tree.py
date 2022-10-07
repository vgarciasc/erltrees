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

    def act(self, state):
        self.visits += 1

        if self.is_leaf():
            return self.label
        
        if state[self.attribute] <= self.threshold:
            return self.left.act(state)
        else:
            return self.right.act(state)
    
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
    
    def get_node_list(self, get_inner=True, get_leaf=True):
        stack = [self]
        output = []

        while len(stack) > 0:
            node = stack.pop()
            if (node.is_leaf() and get_leaf) or (not node.is_leaf() and get_inner):
                output.append(node)

            if not node.is_leaf():
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)

        return output

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