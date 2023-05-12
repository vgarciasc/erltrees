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
        if self.config['name'] == 'MountainCar-v0':
            state = np.clip((state - np.array([-0.4374382, 0.00898582])) / np.sqrt(np.array([0.15740141, 0.00069881]) + 1e-08), -10, 10)
        return self.model.predict(state, deterministic=True)[0]

    def batch_predict(self, states):
        return [self.act(state) for state in states]

    def load_model(self, filename):
        env = self.config["maker"]()
        self.model = PPO.load(filename, env=env)

if __name__ == "__main__":
    # Load environment, model file, and number of episodes from command line
    parser = argparse.ArgumentParser(description='Stable Baselines Load')
    parser.add_argument('-t', '--task', help="Which task to run?", required=True)
    parser.add_argument('-f', '--file', help="Which file to load?", required=True)
    parser.add_argument('-e', '--episodes', help="How many episodes to run?", required=False, default=10, type=int)
    parser.add_argument('-s', '--task_solution_threshold', help="What is the threshold for a task to be considered solved?", required=False, default=0, type=int)
    args = vars(parser.parse_args())

    # Load config
    config = get_config(args['task'])
    agent = PPOAgent(config)
    agent.load_model(args['file'])

    # Print rewards
    total_rewards = []

    env = config['maker']()
    env.reset()

    for ep in range(args['episodes']):
        print(f"Episode {ep}")
        obs = env.reset()
        done = False
        total_rewards.append(0)

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_rewards[-1] += reward

        print(f"Total reward: {total_rewards[-1]}")

    print(f"Average reward: {np.mean(total_rewards)} +/- {np.std(total_rewards)}")
    print(f"Success rate: {np.mean([1 if r > args['task_solution_threshold'] else 0 for r in total_rewards])}")

