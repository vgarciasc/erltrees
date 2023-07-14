import argparse
import json
from datetime import datetime

from stable_baselines3 import PPO
from baseline_policies import NullAgent, ExpertAgent, TreeAgent
from sb3_wrapper import GymDssatWrapper
from stable_baselines3.common.monitor import Monitor
from gym_dssat_pdi.envs.utils import utils
from erltrees.reevaluator import get_trees_from_logfile
from copy import deepcopy
import gym
import pickle
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import time
import joblib

def evaluate_episodes(agent, eval_args, episodes=10):
    env = GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_args))
    try:
        total_rewards = []
        actions_taken = []
        for _ in range(episodes):
            done = False
            observation = env.reset()
            total_rewards.append([])
            actions_taken.append([])
            while not done:
                action = agent.predict(observation, deterministic=True)[0]
                observation, reward, done, _ = env.step(action=action)
                total_rewards[-1].append(reward)
                actions_taken[-1].append(action[0])
    finally:
        env.close()
    return total_rewards, actions_taken

def evaluate_par(agent, eval_args, n_episodes=100, n_jobs=2):
    episodes_partition = [n_episodes // n_jobs for _ in range(n_jobs)]
    episodes_partition[-1] += n_episodes % n_jobs

    results = joblib.parallel.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.parallel.delayed(evaluate_episodes)(agent, eval_args, partition) for partition in episodes_partition)
    all_histories = [{"rewards": r, "actions": a} for rewards, actions in results for r, a in zip(rewards, actions)]
    return all_histories

def evaluate_seq(agent, eval_args, n_episodes=100):
    # Create eval env
    source_env = gym.make('GymDssatPdi-v0', **eval_args)
    env = GymDssatWrapper(source_env)
    all_histories = []
    try:
        tik = time.perf_counter()
        for _ in range(n_episodes):
            done = False
            observation = env.reset()
            total_rewards = []
            actions_taken = []
            while not done:
                action = agent.predict(observation, deterministic=True)[0]
                observation, reward, done, _ = env.step(action=action)
                total_rewards.append(reward)
                actions_taken.append(action[0])
            all_histories.append({"rewards": total_rewards, "actions": np.array(actions_taken)})
        tok = time.perf_counter()
        print(f"Time taken for {n_episodes} episodes: {tok - tik}")
    finally:
        env.close()
    return all_histories

def evaluate(agent, eval_args, n_episodes=100, n_jobs=1):
    if n_jobs == 1:
        return evaluate_seq(agent, eval_args, n_episodes)
    else:
        return evaluate_par(agent, eval_args, n_episodes, n_jobs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GymDSSAT')
    parser.add_argument('--ppo_file', required=False, default="output/fertilization/final_model.zip", type=str)
    parser.add_argument('--tree_file', required=False, default=None, type=str)
    parser.add_argument('--tree_file_type', required=False, default="long", type=str)
    parser.add_argument('--tree_idx', required=False, default=0, type=int)
    parser.add_argument('--episodes', required=False, default=100, type=int)
    parser.add_argument('--experts_file', required=False, default="expert_and_null.pkl", type=str)
    parser.add_argument('--task_solution_threshold', required=False, default=50, type=float)
    parser.add_argument('--load_experts', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_experts', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--skip_ppo', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--denormalize_tree', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_jobs', required=False, default=1, type=int)
    args = vars(parser.parse_args())

    env_args = {
        'mode': 'fertilization',
        # 'mode': 'irrigation',
        # 'seed': 123,
        'random_weather': True,
        # 'random_weather': False,
        'evaluation': True,  # isolated seeds for weather generation
    }

    print(f'###########################\n## MODE: {env_args["mode"]} ##\n###########################')

    assert os.path.exists(args["ppo_file"])

    source_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env = Monitor(GymDssatWrapper(source_env))
    n_episodes = args["episodes"]

    if args["tree_file"] is None:
        tree_str = None
    else:
        if args["tree_file_type"] == "long":
            tree_str = get_trees_from_logfile(args["tree_file"])[args["tree_idx"]]
            print(f"Using tree #{args['tree_idx']} from {args['tree_file']}")
        else:
            with open(args["tree_file"], 'r') as f:
                tree_str = json.load(f)[args["tree_idx"]]
            print(f"Using tree from {args['tree_file']}")

    try:
        agents = {
            'tree': TreeAgent(env, tree_str, args['denormalize_tree']),
            'null': NullAgent(env),
            'expert': ExpertAgent(env),
            'ppo': PPO.load(args["ppo_file"]),
        }

        if args["tree_file"] is None:
            agents.pop('tree')
        else:
            print(f"Loaded tree agent:")
            print(f"{agents['tree'].tree}")

        all_histories = {}
        if args["load_experts"]:
            with open(args["experts_file"], 'rb') as f:
                all_histories = pickle.load(f)

        for agent_name in [*agents]:
            if args["load_experts"] and agent_name in ["null", "expert"]:
                continue

            if args["skip_ppo"] and agent_name == "ppo":
                continue

            agent = agents[agent_name]
            print(f'Evaluating {agent_name} agent...')
            tik = time.perf_counter()
            histories = evaluate(agent=agent, eval_args=env_args, n_episodes=n_episodes, n_jobs=args["n_jobs"])
            tok = time.perf_counter()
            print(f'\tEvaluation took {tok - tik:.2f} seconds')
            histories = utils.transpose_dicts(histories)
            all_histories[agent_name] = histories
            print(f"\tAverage reward: {np.mean([np.sum(x) for x in histories['rewards']]):.3f} ± "
                  f"{np.std([np.sum(x) for x in histories['rewards']]):.3f}")
            success_rate = np.mean([1 if np.sum(x) > args["task_solution_threshold"] else 0 for x in histories['rewards']])
            print(f"\tSuccess rate: {success_rate:.3f}")

        saving_path = f'./output/{env_args["mode"]}/evaluation_histories.pkl'
        with open(saving_path, 'wb') as handle:
            pickle.dump(all_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        env.close()

    if args["save_experts"]:
        with open(args["experts_file"], 'wb') as f:
            pickle.dump(all_histories, f)

    with open(os.path.join(os.path.dirname(args["ppo_file"]), f"histories_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(all_histories, f)

    # plt.style.use('ggplot')
    for agent_name in [*agents]:
        R = all_histories[agent_name]['rewards']
        rewards = np.array([np.pad(r[:-1], pad_width=(0, 180 - len(r[:-1])), constant_values=0) for r in R])
        cum_rewards = np.cumsum(rewards, axis=1)
        mean_rewards = np.mean(cum_rewards, axis=0)
        std_rewards = np.std(cum_rewards, axis=0)
        plt.plot(range(180), mean_rewards, label=agent_name)
        plt.fill_between(range(180), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    figpath = os.path.join(os.path.dirname(args["ppo_file"]), f"evaluation_{timestamp}.png")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.savefig(figpath)
    print(f"Saved image to {figpath}")
