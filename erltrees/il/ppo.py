from tabnanny import verbose
import gym
# import gym_snake
import random
import pdb
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gc import collect
from rich import print
from collections import deque
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1
from keras.optimizers import Adam

from erltrees.rl.configs import get_config
from erltrees.il.distilled_tree import DistilledTree
import erltrees.rl.utils as rl

from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, config):
        self.n_attributes = config["n_attributes"]
        self.n_actions = config["n_actions"]
        self.config = config

        self.model = None

    def act(self, state):
        state = np.clip((state - np.array([-0.4374382, 0.00898582])) / np.sqrt(np.array([0.15740141, 0.00069881]) + 1e-08), -10, 10)
        return self.model.predict(state, deterministic=True)[0]

    def batch_predict(self, states):
        return [self.act(state) for state in states]

    def load_model(self, filename):
        env = gym.make(self.config["name"])
        self.model = PPO.load(filename, env=env)