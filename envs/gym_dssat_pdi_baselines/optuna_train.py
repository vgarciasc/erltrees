import os
import pickle as pkl
import random
import sys
import time
import gym
from pprint import pprint
import argparse
import numpy as np

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from optuna_utils import *
from sb3_wrapper import GymDssatWrapper


def objective(trial: optuna.Trial) -> float:
    attrs = trial.study.user_attrs
    path = f"{attrs['log_saving_path']}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    params = sample_ppo_params(trial)

    env_args = {
        'log_saving_path': './logs/dssat_pdi.log',
        'mode': 'fertilization',
        # 'seed': attrs["seed"],
        'random_weather': attrs["random_weather"],
    }

    # eval_env_args = {**env_args, 'seed': 345}
    eval_env_args = {**env_args}
    env = GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **eval_env_args))
    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=345, verbose=0, tensorboard_log=path, **params)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=attrs['max_no_improvement_evals'],
        min_evals=attrs['min_evals'], verbose=1)

    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=attrs["n_eval_episodes"], eval_freq=attrs["eval_freq"],
        deterministic=False,
        callback_after_eval=stop_callback
    )

    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(attrs["learning_timesteps"], callback=eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Baselines Load')
    parser.add_argument('-t', '--timesteps', help="How many timesteps in each epoch?", required=False, default=10000,
                        type=int)
    parser.add_argument("--simul_name", help="Name of the simulation", required=False, default="test", type=str)
    parser.add_argument('--output_path', help="Where to save the results?", required=False, default=".", type=str)
    parser.add_argument('--log_path', help="Where to log the results?", required=False, default=".", type=str)
    parser.add_argument('--n_jobs', help="Number of jobs to run in parallel", required=False, default=4, type=int)
    parser.add_argument('--max_no_improvement_evals', help="How many evaluations to wait for improvement?",
                        required=False, default=10, type=int)
    parser.add_argument('--eval_freq', help="How often to evaluate the model?", required=False, default=1000, type=int)
    parser.add_argument('--min_evals', help="How many evaluations to run?", required=False, default=10, type=int)
    parser.add_argument('--n_eval_episodes', help="How many episodes to evaluate?", required=False, default=10, type=int)
    parser.add_argument('--random_weather', help="Should use random weather?", required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_trials', help="Number of trials to run", required=False, default=128, type=int)
    parser.add_argument('--verbose', help="Print more information?", required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=args["timesteps"] // 5)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    output_path = os.path.join(args["output_path"], args["simul_name"])
    log_path = os.path.join(args["log_path"], args["simul_name"])

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    study.set_user_attr("n_eval_episodes", args["n_eval_episodes"])
    study.set_user_attr("eval_freq", args["eval_freq"])
    study.set_user_attr("max_no_improvement_evals", args["max_no_improvement_evals"])
    study.set_user_attr("min_evals", args["min_evals"])
    study.set_user_attr("random_weather", args["random_weather"])
    study.set_user_attr("log_saving_path", log_path)
    study.set_user_attr("output_path", output_path)
    study.set_user_attr("learning_timesteps", args["timesteps"])

    try:
        study.optimize(objective, n_jobs=args["n_jobs"], n_trials=args["n_trials"])
    except KeyboardInterrupt:
        pass

    trial = study.best_trial
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {trial.number}")
    print(f"Value: {trial.value}")

    print("Params: ")
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")

    study.trials_dataframe().to_csv(f"{args['output_path']}/report.csv")

    print(f"Saving study to '{args['output_path']}/study.pkl'")
    with open(f"{args['output_path']}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)
