from cyclesgym.envs import Corn
from cyclesgym.envs.weather_generator import FixedWeatherGenerator
from cyclesgym.utils.paths import CYCLES_PATH  # Path to Cycles directory
from cyclesgym.managers import OperationManager
from cyclesgym.policies.dummy_policies import OpenLoopPolicy

import matplotlib.pyplot as plt
import numpy as np

from experiments.fertilization.corn_soil_refined import CornSoilRefined

if __name__ == "__main__":
    env = CornSoilRefined(delta=7, maxN=150, n_actions=11,
                          start_year=1980, end_year=1980,
                          sampling_start_year=1980,
                          sampling_end_year=2005,
                          n_weather_samples=100,
                          fixed_weather=True,
                          with_obs_year=False,
                          new_holland=False)

    rewards = []

    for ep in range(10):
        print(f"Episode {ep}")
        done = False
        state = env.reset()
        rewards.append(0)

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            rewards[-1] += reward

    print(f"Mean reward: {np.mean(rewards)} +- {np.std(rewards)}")
