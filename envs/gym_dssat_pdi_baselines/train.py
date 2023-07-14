import pdb

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_wrapper import GymDssatWrapper
from gym_dssat_pdi.envs.utils import utils as dssat_utils
import gym

if __name__ == '__main__':
    try:
        for dir in ['./output', './logs']:
            dssat_utils.make_folder(dir)

        # Create environment
        env_args = {
            'log_saving_path': './logs/dssat_pdi.log',
            'mode': 'fertilization',
            # 'mode': 'irrigation',
            'seed': 123,
            'random_weather': True,
        }

        print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

        env = Monitor(GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)))

        # Training arguments for PPO agent
        ppo_args = {
            'seed': 123,  # seed training for reproducibility
            'gamma': 1,
        }

        # Create the agent
        ppo_agent = PPO('MlpPolicy', env, **ppo_args)

        # path to save best model found
        path = f'./output/{env_args["mode"]}'

        # eval callback
        eval_freq = 1000
        eval_env_args = {**env_args, 'seed': 345}
        eval_env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_env_args)))
        eval_callback = EvalCallback(eval_env,
                                     eval_freq=eval_freq,
                                     best_model_save_path=f'{path}',
                                     deterministic=True,
                                     n_eval_episodes=10)

        # Train
        # total_timesteps = 500_000
        total_timesteps = 1_000_000
        print('Training PPO agent...')
        ppo_agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
        ppo_agent.save(f'{path}/final_model')
        print('Training done')
        pdb.set_trace()
    finally:
        env.close()