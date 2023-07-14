from timeit import default_timer as timer
import gym
import numpy as np
import sys
import warnings
import pdb

if __name__ == '__main__':
    # humanoidstandup = gym.make('HumanoidStandup-v2')
    # humanoidstandup.reset()
    env_args = {
        'mode': 'fertilization',
        'seed': 123,
    }
    gymDSSAT = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    gymDSSAT.reset()
    repetitions = 1000
    steps = 100

    # print(f'Measuring step time for {steps} time steps, {repetitions} repetitions for HumanoidStandup-v2')
    # humanoidstandup_times = []
    # for repetition in range(repetitions):
    #     if (repetition + 1) % 100 == 0:
    #         print(f'repetition {repetition+1}/{repetitions}')
    #     done = True
    #     c = 0
    #     while done:
    #         c += 1
    #         episode_steps = []
    #         for _ in range(steps):
    #             action = humanoidstandup.action_space.sample()
    #             start = timer()
    #             _, _, done, _ = humanoidstandup.step(action)
    #             end = timer()
    #             if done:
    #                 warnings.warn('Warning: the episode has early ended for HumanoidStandup-v2')
    #             episode_steps.append(end-start)  # the time difference is in seconds by default
    #         if c == 100:
    #             warnings.warn('Too many incomplete episodes')
    #             sys.exit()
    #         humanoidstandup.reset()
    #     humanoidstandup_times.extend(episode_steps)
    # humanoidstandup.close()

    # print(f'\nHumanoidStandup-v2 mean step time {1000 * np.mean(humanoidstandup_times)} milliseconds, '
    #       f'std {1000 * np.std(humanoidstandup_times, ddof=1)} milliseconds\n')

    print(f'Measuring step time for {steps} time steps, {repetitions} repetitions for GymDssatPdi-v0')
    gymDSSAT_times = []
    for repetition in range(repetitions):
        if (repetition + 1) % 100 == 0:
            print(f'repetition {repetition+1}/{repetitions}')
        done = True
        c = 0
        while done:
            c += 1
            episode_steps = []
            for _ in range(steps):
                action = dict(gymDSSAT.action_space.sample())
                action = {key: action[key].item() for key in [*action]}
                start = timer()
                _, _, done, _ = gymDSSAT.step(action)
                end = timer()
                if done:
                    warnings.warn('Warning: the episode has early ended for gymDSSAT')
                episode_steps.append(end-start)
            if c == 100:
                warnings.warn('Too many incomplete episodes')
                sys.exit()
            gymDSSAT.reset()
        gymDSSAT_times.extend(episode_steps)
    gymDSSAT.close()

    print(f'\ngymDSSAT mean step time {1000 * np.mean(gymDSSAT_times)} milliseconds,'
          f' std {1000 * np.std(gymDSSAT_times, ddof=1)} milliseconds\n')