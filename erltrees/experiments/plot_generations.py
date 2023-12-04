import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rc
import matplotlib.gridspec as gridspec

# plt.rcParams.update({'font.size': 12})
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

    print(f"Identified {len(solutions)} simulations.")
    solutions = [np.array(sol) for sol in solutions if len(sol) == len(solutions[0])]
    print(f"Selected {len(solutions)} simulations, after removing incomplete ones.")

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
    ax.set_ylabel('Average reward of best solution')

    ax = plt.subplot(1, 2, 2)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Tree size of best solution')

    # Show the plot
    plt.suptitle(title)
    plt.legend()
    plt.show()

def generic_plot_2(title, files, algos, solution_threshold):
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 1, hspace=0.0, height_ratios=[0.1, 1], wspace=0.5)
    gs_legend = gs[0].subgridspec(1, 1)
    gs_figs = gs[1].subgridspec(2, 2, hspace=0.24)

    x = range(200) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    ax = plt.subplot(gs_figs[0, 0])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # Set the x-axis label
    # ax.legend()
    ax.set_xlabel('(a)')
    ax.set_ylabel('Avg. fitness of best solution')

    ax = plt.subplot(gs_figs[0, 1])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    # put the legend where the title is, remove the frame and put it horizontally
    ax.set_xlabel('(b)')
    ax.set_ylabel('Average reward of best solution')
    handles, labels = ax.get_legend_handles_labels()

    ax = plt.subplot(gs_figs[1, 0])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 2]
        sol_std = np.std(sol, axis=0)[:, 2]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # ax.legend()
    ax.set_xlabel('Generations\n(c)')
    ax.set_ylabel('Avg. stdev. of reward of best solution')

    ax = plt.subplot(gs_figs[1, 1])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # ax.legend()
    ax.set_xlabel('Generations\n(d)')
    ax.set_ylabel('Average tree size of best solution')

    ax = plt.subplot(gs_legend[0, :])
    ax.legend(handles, labels, loc='center', ncol=2, frameon=False, prop={'size': 14})
    ax.axis('off')

    # Show the plot
    # plt.suptitle(title)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.093, bottom=0.112, right=0.983, top=1, wspace=0.17, hspace=0.373)
    # plt.tight_layout()
    plt.savefig(f"fig_MENS_{title}.pdf", dpi=100)
    plt.show()

def generic_plot_3(title, files, algos, solution_threshold, n_gens=50, is_dssat=False):
    plt.rcParams.update({'font.size': 12})

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 1, wspace=-0, height_ratios=[0.1, 1])
    gs_legend = gs[0].subgridspec(1, 1)
    gs_figs = gs[1].subgridspec(1, 3, wspace=0.25)

    x = range(n_gens) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    ax = plt.subplot(gs_figs[0, 0])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # Set the x-axis label
    ax.set_xlabel('Generations\n\n(a)')
    ax.set_ylabel('Average fitness of best solution')

    if is_dssat:
        ax.set_ylim(-125, 50)

    ax = plt.subplot(gs_figs[0, 1])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Generations\n\n(b)')
    ax.set_ylabel('Average reward of best solution')

    if is_dssat:
        ax.set_ylim(-100, 75)

    ax = plt.subplot(gs_figs[0, 2])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.set_xlabel('Generations\n(c)')
    ax.set_ylabel('Tree size of best solution')

    ax = plt.subplot(gs_legend[0, :])
    ax.legend(handles, labels, loc='center', ncol=2, frameon=False, prop={'size': 14})
    ax.axis('off')

    # Show the plot
    # plt.suptitle(title)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(wspace=0.255)
    plt.tight_layout()
    plt.savefig(f"fig_MENS_{title}.pdf", dpi=100)
    plt.show()

def generic_plot_4(title, files, algos, solution_threshold, n_gens=50, is_dssat=False):
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(10, 9))
    spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig, height_ratios=[0.01, 0.4995, 0.4995], hspace=0.3, wspace=0.8)

    x = range(n_gens) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    ax = fig.add_subplot(spec[1, 0:2])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # Set the x-axis label
    ax.set_xlabel('(a)')
    ax.set_ylabel('Average fitness of best solution')

    if is_dssat:
        ax.set_ylim(-125, 50)

    ax = fig.add_subplot(spec[1, 2:4])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('(b)')
    ax.set_ylabel('Average reward of best solution')

    if is_dssat:
        ax.set_ylim(-100, 75)

    ax = fig.add_subplot(spec[2, 1:3])
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.set_xlabel('Generations\n(c)')
    ax.set_ylabel('Tree size of best solution')

    ax = fig.add_subplot(spec[0, 1:3])
    ax.legend(handles, labels, loc='center', ncol=2, frameon=False, prop={'size': 14})
    ax.axis('off')

    # Show the plot
    # plt.suptitle(title)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.09, bottom=0.118, right=0.973, top=0.947, wspace=0.555, hspace=0.2)
    # plt.tight_layout()
    plt.savefig(f"fig_MENS_{title}.pdf", dpi=100)
    plt.show()

def generic_plot_separate(title, files, algos, solution_threshold, n_gens=50, is_dssat=False):
    x = range(n_gens) # Number of generations
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    figsize = (6, 4)

    ############
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 2]
        sol_std = np.std(sol, axis=0)[:, 2]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # Set the x-axis label
    ax.legend()
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average stdev. of reward of best solution')

    if is_dssat:
        ax.set_ylim(-125, 50)

    plt.tight_layout()
    plt.savefig(f"fig_{title}_std.pdf", dpi=200)

    ############
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 0]
        sol_std = np.std(sol, axis=0)[:, 0]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    # Set the x-axis label
    ax.legend(loc="lower right")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness of best solution')

    if is_dssat:
        ax.set_ylim(-125, 50)

    plt.tight_layout()
    plt.savefig(f"fig_{title}_fitness.pdf", dpi=200)

    ############
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 1]
        sol_std = np.std(sol, axis=0)[:, 1]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.plot(x, np.ones(len(x)) * solution_threshold, label="Solution threshold", color="black", linestyle="dashed")

    ax.legend(loc="lower right")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average reward of best solution')

    if is_dssat:
        ax.set_ylim(-100, 75)

    plt.tight_layout()
    plt.savefig(f"fig_{title}_reward.pdf", dpi=200)

    ###########
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(color='0.9', linestyle='--', linewidth=1, zorder=-5)

    for i, file in enumerate(files):
        sol = parse_file(file)
        sol_mean = np.mean(sol, axis=0)[:, 3]
        sol_std = np.std(sol, axis=0)[:, 3]
        ax.plot(x, sol_mean, label=algos[i], color=colors[i])
        l = ax.fill_between(x, sol_mean - sol_std, sol_mean + sol_std, alpha=0.2, color=colors[i])
        l.set_linewidth(0.0)

    ax.legend()
    ax.set_xlabel('Generations')
    ax.set_ylabel('Tree size of best solution')

    plt.tight_layout()
    plt.savefig(f"fig_{title}_treesize.pdf", dpi=200)

    print(f"Finished {title}")

if __name__ == '__main__':
    # generic_plot_separate("Cartpole", ["../../../CRO_DT_RL/results/tmp_files/cartpole_CRO__2023_03_23-15_22_26_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/cartpole_IL-CRO__2023_03_30-16_39_00_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/cartpole_IL-RP-CRO__2023_03_30-16_38_59_tmp.txt"],
    #                ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], 495, n_gens=200)
    #
    # generic_plot_separate("Mountain Car", ["../../../CRO_DT_RL/results/tmp_files/mountain-car_CRO__2023_02_08-11_08_34_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/mountain-car_IL-CRO__2023_03_21-13_15_47_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/mountain-car_IL-RP-CRO__2023_03_24-12_15_07_tmp.txt"],
    #                ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], -110, n_gens=200)
    #
    # generic_plot_separate("Lunar Lander", ["../../../CRO_DT_RL/results/tmp_files/lunarlander_CRO__2023_02_27-18_53_54_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/lunarlander_IL-CRO__2023_03_02-11_35_43_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/lunarlander_IL-RP-CRO__2023_02_17-12_56_21_tmp.txt"],
    #                ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], 200, n_gens=200)
    #
    # generic_plot_separate("Maize Fertilization", ["../../../CRO_DT_RL/results/tmp_files/dssat_CRO_2023_06_07-11_56_31_FULL_tmp.txt",
    #     "../../../CRO_DT_RL/results/tmp_files/dssat_CRO-IL_2023_06_02-22_02_28_FULL_tmp.txt",
    #     "../../../CRO_DT_RL_pycro-sl/final_results/original/dssat_CRO-IL-RP_tmp.txt"],
    #                ["CRO-DT-RL (R)", "CRO-DT-RL (IL)", "CRO-DT-RL (P)"], 60, n_gens=50, is_dssat=True)

    generic_plot_4("Cartpole", ["../../../CRO_DT_RL/results/tmp_files/cartpole_CRO__2023_03_23-15_22_26_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/cartpole_IL-CRO__2023_03_30-16_39_00_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/cartpole_IL-RP-CRO__2023_03_30-16_38_59_tmp.txt"],
                   ["MENS-DT-RL (R)", "MENS-DT-RL (IL)", "MENS-DT-RL (P)"], 495, n_gens=200)

    generic_plot_2("Mountain Car", ["../../../CRO_DT_RL/results/tmp_files/mountain-car_CRO__2023_02_08-11_08_34_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/mountain-car_IL-CRO__2023_03_21-13_15_47_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/mountain-car_IL-RP-CRO__2023_03_24-12_15_07_tmp.txt"],
                   ["MENS-DT-RL (R)", "MENS-DT-RL (IL)", "MENS-DT-RL (P)"], -110)

    generic_plot_2("Lunar Lander", ["../../../CRO_DT_RL/results/tmp_files/lunarlander_CRO__2023_02_27-18_53_54_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/lunarlander_IL-CRO__2023_03_02-11_35_43_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/lunarlander_IL-RP-CRO__2023_02_17-12_56_21_tmp.txt"],
                   ["MENS-DT-RL (R)", "MENS-DT-RL (IL)", "MENS-DT-RL (P)"], 200)

    generic_plot_4("Maize Fertilization", ["../../../CRO_DT_RL/results/tmp_files/dssat_CRO_2023_06_07-11_56_31_FULL_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/dssat_CRO-IL_2023_06_02-22_02_28_FULL_tmp.txt",
        "../../../CRO_DT_RL/results/tmp_files/dssat_CRO-IL-RP_tmp.txt"],
                   ["MENS-DT-RL (R)", "MENS-DT-RL (IL)", "MENS-DT-RL (P)"], 30, is_dssat=True)