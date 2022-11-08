import gym
import pdb
import pickle
import numpy as np

from rich import print
from erltrees.io import printv

def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []

    printv("Collecting dataset from model.", verbose)
    for i in range(episodes):
        if i % 10 == 0:
            printv(f"{i} / {episodes} episodes... |D| = {len(X)}.", verbose)

        state = env.reset()
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, done, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y

def label_dataset_with_model(model, X):
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