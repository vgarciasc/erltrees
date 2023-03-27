from copy import deepcopy
import math
import time
import numpy as np
import pdb
import matplotlib.pyplot as plt
from rich import print

import erltrees.trees.tree as tree
import erltrees.rl.utils as rl
import erltrees.rl.configs as configs
import erltrees.io as io

class Individual(tree.TreeNode):
    def __init__(self, fitness, reward, std_reward, success_rate, **kwargs):
        super(Individual, self).__init__(**kwargs)

        self.fitness = fitness
        self.reward = reward
        self.std_reward = std_reward
        self.success_rate = success_rate

    def get_random_node(self, get_inners=True, get_leaves=True):
        node_list = self.get_node_list(get_inners, get_leaves)

        if len(node_list) <= 1:
            return self

        return np.random.choice(node_list)

    def generate_random_node(config):
        attribute = np.random.randint(config["n_attributes"])
        threshold = np.random.uniform(-1, 1)
        label = np.random.randint(config["n_actions"])

        return Individual(fitness=None, reward=None, std_reward=None, success_rate=None,
                          config=config, attribute=attribute,
                          threshold=threshold, label=label)

    def generate_random_tree(config, depth=2):
        node = Individual.generate_random_node(config)

        if depth > 0:
            node.left = Individual.generate_random_tree(config, depth - 1)
            node.right = Individual.generate_random_tree(config, depth - 1)
            node.left.parent = node
            node.right.parent = node

        return node

    def mutate_attribute(self, verbose=False):
        io.printv("Mutating attribute...", verbose)

        new_attribute = self.attribute
        while new_attribute == self.attribute:
            new_attribute = np.random.randint(self.config["n_attributes"])
        self.attribute = new_attribute

    def mutate_threshold(self, sigma=1, verbose=False):
        io.printv("Mutating threshold...", verbose)

        attr_name, attr_type, (min_val, max_val) = self.config["attributes"][self.attribute]
        if attr_type == "continuous":
            if type(sigma) == np.ndarray or type(sigma) == list:
                sigma = sigma[self.attribute]

            self.threshold += np.random.normal(0, 1) * sigma
            self.threshold = np.clip(self.threshold, min_val, max_val)
        elif attr_type == "binary":
            self.threshold = 0.5

    def mutate_label(self, verbose=False):
        io.printv("Mutating label...", verbose)

        gen_label = lambda : np.random.randint(self.config["n_actions"])

        if self.is_root():
            new_label = gen_label()
        else:
            if self.is_left_child() and self.parent.right.is_leaf():
                other_label = self.parent.right.label
            elif self.is_right_child() and self.parent.left.is_leaf():
                other_label = self.parent.left.label
            else: #elif self.is_root():
                other_label = self.label

            new_label = gen_label()
            while new_label == other_label:
                new_label = gen_label()

        self.label = new_label

    def mutate_is_leaf(self, verbose=False):
        io.printv("Mutating is leaf...", verbose)

        if not self.is_leaf():
            self.left = None
            self.right = None
        else:
            self.left = Individual.generate_random_node(self.config)
            self.right = Individual.generate_random_node(self.config)
            self.left.parent = self
            self.right.parent = self

            labels = np.random.choice(
                range(0, self.config["n_actions"]),
                size=2, replace=False)
            self.left.label = labels[0]
            self.right.label = labels[1]

    def mutate_add_inner_node(self, verbose=False):
        io.printv("Adding inner node...", verbose)

        if self.is_root():
            return

        new_stump = Individual.generate_random_tree(self.config, depth=1)

        if self.is_left_child():
            self.parent.left = new_stump
        elif self.is_right_child():
            self.parent.right = new_stump

        new_stump.parent = self.parent
        self.parent = new_stump

        if np.random.uniform() <= 0.5:
            new_stump.left = self
        else:
            new_stump.right = self

    def mutate_truncate(self, verbose=False):
        io.printv("Truncating inner node...", verbose)

        new_node = Individual.generate_random_node(self.config)
        self.right = new_node
        self.left.parent = self
        self.right.parent = self

        labels = np.random.choice(
            range(0, self.config["n_actions"]),
            size=2, replace=False)
        self.left.label = labels[0]
        self.right.label = labels[1]

    def mutate_truncate_dx(self, verbose=False):
        io.printv("Truncating inner node...", verbose)

        new_node = Individual.generate_random_node(self.config)

        if np.random.uniform() < 0.5:
            self.right = new_node
        else:
            self.left = new_node

        self.left.parent = self
        self.right.parent = self

        labels = np.random.choice(
            range(0, self.config["n_actions"]),
            size=2, replace=False)
        self.left.label = labels[0]
        self.right.label = labels[1]

    def replace_child(self, verbose=False):
        io.printv("Replacing child...", verbose)

        if np.random.uniform() < 0.5:
            self.left.cut_parent()
        else:
            self.right.cut_parent()

    def cut_parent(self, verbose=False):
        io.printv("Cutting parent...", verbose)

        if self.is_root() or self.parent.is_root():
            return

        if self.parent.is_left_child():
            self.parent.parent.left = self
        elif self.parent.is_right_child():
            self.parent.parent.right = self

        self.parent = self.parent.parent

    def prune_by_visits(self, threshold=1):
        if self.is_leaf():
            return self

        if self.left.visits < threshold:
            self.right.prune_by_visits(threshold)
            self.right.cut_parent()

            if self.is_root():
                return self.right
        elif self.right.visits < threshold:
            self.left.cut_parent()
            self.left.prune_by_visits(threshold)

            if self.is_root():
                return self.left
        else:
            self.left.prune_by_visits(threshold)
            self.right.prune_by_visits(threshold)

            if self.is_root():
                return self

    def crossover(parent_a, parent_b):
        parent_a = parent_a.copy()
        parent_b = parent_b.copy()

        node_a = parent_a.get_random_node(get_leaves=False)
        node_b = parent_b.get_random_node(get_leaves=False)

        parent_a.replace_node(node_a, node_b)
        parent_b.replace_node(node_b, node_a)

        return parent_a, parent_b

    def copy(self):
        if self.is_leaf():
            return Individual(fitness=self.fitness,
                              reward=self.reward,
                              std_reward=self.std_reward,
                              success_rate=self.success_rate,
                              config=self.config,
                              attribute=self.attribute,
                              threshold=self.threshold,
                              label=self.label)
        else:
            new_left = self.left.copy()
            new_right = self.right.copy()

            new_node = Individual(fitness=self.fitness,
                                  reward=self.reward,
                                  std_reward=self.std_reward,
                                  success_rate=self.success_rate,
                                  config=self.config,
                                  attribute=self.attribute,
                                  threshold=self.threshold,
                                  label=self.label,
                                  left=new_left,
                                  right=new_right)

            new_left.parent = new_node
            new_right.parent = new_node

            return new_node

    def read_from_string(config, string):
        actions = [a.lower() for a in config['actions']]
        attributes = [name.lower() for name, _, _ in config['attributes']]

        lines = [line.strip() for line in string.split("\n")]

        parents = [None for _ in lines]
        child_count = [0 for _ in lines]

        for line in lines[1:]:
            depth = line.rindex("- ") + 1

            content = line[depth:].strip()

            parent = parents[depth - 1] if depth > 1 else None
            is_left = (child_count[depth - 1] == 0) if depth > 1 else None

            is_leaf = "<=" not in content

            if not is_leaf:
                attribute, threshold = content.split(" <= ")

                attribute = attributes.index(attribute.lower())
                threshold = float(threshold)

                node = Individual(fitness=-1, reward=-1, std_reward=-1, success_rate=-1,
                    config=config, attribute=attribute, threshold=threshold, label=0,
                    left=None, right=None, parent=parent)

            if is_leaf:
                label = actions.index(content.lower())

                node = Individual(fitness=-1, reward=-1, std_reward=-1, success_rate=-1,
                    config=config, attribute=0, threshold=0, label=label,
                    left=None, right=None, parent=parent)

            if parent:
                if is_left:
                    parent.left = node
                else:
                    parent.right = node
            else:
                root = node

            parents[depth] = node
            child_count[depth] = 0
            child_count[depth - 1] += 1

        return root