import pdb
import time
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from gym_crop.envs.fertilization_env import FertilizationEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ppo = PPO.load("results/cropgym/interval1_beta_10_0.zip")
    rewards = []

    for year in [1984, 1994, 2004, 2014]:
        env = FertilizationEnv(beta=10, intervention_interval=1,
                               fixed_year=year, fixed_location=[52, 5.5])
        env = DummyVecEnv([lambda: env])
        # env = VecNormalize.load("results/cropgym/interval1_beta_10_0.pkl", env)

        state = env.reset()
        step = 0
        done = False
        actions = []
        rewards.append([])

        tik = time.perf_counter()
        while not done:
            # action = [env.action_space.sample()]

            if step == 60 or step == 88 or step == 123:
                # print(env.date)
                action = [3]
            else:
                action = [0]

            # if state[0][6] <= 0.5:
            #     action = [6]
            # else:
            #     action = [0]

            # action = ppo.predict(state, deterministic=True)

            state, reward, done, _ = env.step(action)
            actions.append(action[0])
            rewards[-1].append(reward[0])
            step += 1
        tok = time.perf_counter()

        print(f"Year {year}:\t Reward: {np.sum(rewards[-1]):.5f}\t Steps: {len(rewards[-1])}\t Elapsed time: {(tok - tik):.3f} seconds")

        plt.subplots(2, 1, figsize=(10, 5))

        ax = plt.subplot(2, 1, 1)
        ax.plot(range(len(rewards[-1])), rewards[-1])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")

        ax = plt.subplot(2, 1, 2)
        ax.plot(range(len(actions)), actions)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Action")

        plt.show()
