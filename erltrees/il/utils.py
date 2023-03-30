import gym
import pdb
import pickle
import numpy as np

from rich import print

from erltrees.il.ppo import PPOAgent
from erltrees.io import printv

def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []
    total_rewards = []

    printv("Collecting dataset from model.", verbose)
    for i in range(episodes):
        if i % 10 == 0:
            printv(f"{i} / {episodes} episodes... |D| = {len(X)}.", verbose)

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y, total_rewards

def label_dataset_with_model(model, X):
    if type(model) == PPOAgent:
        y = [model.act(x) for x in X]
    else:
        y = model.batch_predict(X)
        y = [np.argmax(q) for q in y]
    return y

def save_dataset(filename, X, y):
    with open(filename, "wb") as f:
        pickle.dump((X, y), f)

def load_dataset(filename):
    with open(filename, "rb") as f:
        X, y = pickle.load(f)
    return X, y