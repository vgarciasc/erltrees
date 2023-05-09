import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import pandas as pd

ALPHA = 1

def parse_file(filename):
    solutions = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line == "Generation #1 \n":
                solutions.append([])

            if line.startswith("Generation #"):
                tree_strs = lines[i+1].split(" ")
                # Avg Reward, Std Reward, Tree Size, Success Rate
                avg_reward = float(tree_strs[1])
                std_reward = float(tree_strs[3][:-2])
                tree_size = float(tree_strs[5][:-1])
                success_rate = float(tree_strs[8])
                fitness = avg_reward - std_reward - tree_size * ALPHA

                solutions[-1].append((fitness, avg_reward, std_reward, tree_size, success_rate))

    return np.array(solutions)

def generic_plot_1(title, files, algos):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    x = range(200) # Number of generations

    ax = plt.subplot(1, 2, 1)
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    # Set the x-axis label
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average Reward')

    ax = plt.subplot(1, 2, 2)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Tree size')

    # Show the plot
    plt.suptitle(title)
    plt.legend()
    plt.show()

def generic_plot_2(title, files, algos, solution_threshold):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    x = range(200) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    ax = plt.subplot(2, 2, 1)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    # Set the x-axis label
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average best fitness')

    ax = plt.subplot(2, 2, 2)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    ax.set_xlabel('Generations')
    ax.set_ylabel('Average reward of best individual')

    ax = plt.subplot(2, 2, 3)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 2]
        sol_std = np.std(sol, axis=0)[:, 2]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Average stdev. of reward of best individual')

    ax = plt.subplot(2, 2, 4)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Average tree size of best individual')

    # Show the plot
    plt.suptitle(title)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.subplots_adjust(left=0.07, bottom=0.11, right=0.98, top=0.938, wspace=0.214, hspace=0.2)
    plt.show()

def generic_plot_3(title, files, algos, solution_threshold):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    sns.set_theme()

    x = range(200) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    ax = plt.subplot(1, 3, 1)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    # Set the x-axis label
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')

    ax = plt.subplot(1, 3, 2)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    ax.set_xlabel('Generations')
    ax.set_ylabel('Average Reward')

    ax = plt.subplot(1, 3, 3)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Tree size')

    # Show the plot
    plt.suptitle(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(wspace=0.255)
    plt.show()

if __name__ == '__main__':
    generic_plot_3("Cartpole", ["../../../CRO_DT_RL/results/complete/cartpole_CRO__tmp_2023_03_23-15_22_26.txt",
        "../../../CRO_DT_RL/results/complete/cartpole_IL-CRO__tmp_2023_03_30-16_39_00.txt",
        "../../../CRO_DT_RL/results/complete/cartpole_IL-RP-CRO__tmp_2023_03_30-16_38_59.txt"],
                   ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], 495)

    generic_plot_2("Mountain Car", ["../../../CRO_DT_RL/results/complete/mountain-car_CRO__tmp_2023_02_08-11_08_34.txt",
        "../../../CRO_DT_RL/results/complete/mountain-car_IL-CRO__tmp_2023_03_21-13_15_47.txt",
        "../../../CRO_DT_RL/results/complete/mountain-car_IL-RP-CRO__tmp_2023_03_24-12_15_07.txt"],
                   ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], -110)

    generic_plot_2("Lunar Lander", ["../../../CRO_DT_RL/results/complete/lunarlander_CRO__tmp_2023_02_27-18_53_54.txt",
        "../../../CRO_DT_RL/results/complete/lunarlander_IL-CRO__tmp_2023_03_02-11_35_43_GRAFTED.txt",
        "../../../CRO_DT_RL/results/complete/lunarlander_IL-RP-CRO__tmp_2023_02_17-12_56_21.txt"],
                   ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], 200)