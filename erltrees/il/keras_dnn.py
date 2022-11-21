from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents import DQNAgent

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl

import keras
import gym
import argparse
import pdb
import random
import numpy as np

class KerasDNN:
    def __init__(self, config, exploration_rate):
        self.n_attributes = config['n_attributes']
        self.n_actions = config['n_actions']
        self.exploration_rate = exploration_rate

        model = self.build_model(config)
        dqn = DQNAgent(
            model=model, 
            nb_actions=self.n_actions,
            memory=SequentialMemory(limit=50000, window_length=1), 
            nb_steps_warmup=10,
            target_model_update=1e-2, 
            policy=BoltzmannQPolicy())
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        
        self.dqn = dqn
    
    def predict(self, s):
        s = np.reshape(s, (1, self.n_attributes))
        return self.dqn.compute_q_values(s)
    
    def batch_predict(self, X):
        X = np.reshape(X, (len(X), 1, self.n_attributes))
        return self.dqn.compute_batch_q_values(X)
    
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.n_actions)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def build_model(self, env):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.n_attributes,)))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(self.n_actions))
        model.add(Activation('linear'))
        return model
    
    def save(self, filename):
        self.dqn.model.save(filename)
    
    def load_model(self, filename):
        self.dqn.model = keras.models.load_model(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras DNN')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-o','--output_filepath', help='Filepath to save expert', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('--exploration_rate', help='What is the exploration rate?', required=False, default=0.1, type=float)
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # Initialization    
    config = get_config(args['task'])
    env = gym.make(config['name'])
    ann = KerasDNN(config, args["exploration_rate"])

    # Fitting model
    ann.dqn.fit(env, nb_steps=args['iterations'], visualize=False, verbose=2)

    # Evaluating model
    rl.collect_metrics(config, [ann], episodes=100, should_fill_attributes=True)
    print(f"Reward: {ann.reward} +- {ann.std_reward}")
    
    # Saving model
    ann.save(args['output_filepath'])

    # Visualization
    if args['should_visualize']:
        ann.dqn.test(env, nb_episodes=25, visualize=True)