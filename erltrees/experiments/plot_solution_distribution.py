import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import pandas as pd
from cycler import cycler

def parse_file(filename):
    solutions = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("Tree #"):
                tree_strs = line.split(" ")
                solutions.append((float(tree_strs[3]), float(tree_strs[5][:-1]), float(tree_strs[7][:-1]), float(tree_strs[10][:-2])))
    return np.array(solutions)

def attempt_1():
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL_p015_reevaluated.txt")
    plt.scatter(sol[:, 2], sol[:, 3], label="IL", alpha=0.5, marker="^")

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP_reevaluated.txt")
    plt.scatter(sol[:, 2], sol[:, 3], label="IL $\\rightarrow$ RP", alpha=0.5, marker="s")

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_CRO__2023_02_08-11_08_34_reevaluated.txt")
    plt.scatter(sol[:, 2] - 0.3, sol[:, 3], label="CRO-DT-RL", alpha=0.5, marker="*")

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-CRO__2023_03_21-13_15_47_reevaluated.txt")
    plt.scatter(sol[:, 2], sol[:, 3], label="IL $\\rightarrow$ CRO-DT-RL", alpha=0.5, marker="*")

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP-CRO__2023_02_09-14_35_30_reevaluated.txt")
    plt.scatter(sol[:, 2] + 0.3, sol[:, 3], label="IL $\\rightarrow$ RP $\\rightarrow$ CRO-DT-RL", alpha=0.5, marker="*", zorder=-1)

    plt.xlabel("Tree size")
    plt.ylabel("Success rate")
    plt.title("Mountain Car")
    plt.legend()
    plt.show()

def attempt_2():
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    tree_sizes = [3, 5, 7, 9, 11, 13, 15]

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL_p015_reevaluated.txt")
    data = [[x for i, x in enumerate(sol[:, 3]) if tree_size == sol[:, 2][i]] for tree_size in tree_sizes]
    data = [[0] if x == [] else x for x in data]
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP_reevaluated.txt")
    data = [[x for i, x in enumerate(sol[:, 3]) if tree_size == sol[:, 2][i]] for tree_size in tree_sizes]
    data = [[0] if x == [] else x for x in data]
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_CRO__2023_02_08-11_08_34_reevaluated.txt")
    data = [[x for i, x in enumerate(sol[:, 3]) if tree_size == sol[:, 2][i]] for tree_size in tree_sizes]
    data = [[0] if x == [] else x for x in data]
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-CRO__2023_03_21-13_15_47_reevaluated.txt")
    data = [[x for i, x in enumerate(sol[:, 3]) if tree_size == sol[:, 2][i]] for tree_size in tree_sizes]
    data = [[0] if x == [] else x for x in data]
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP-CRO__2023_02_09-14_35_30_reevaluated.txt")
    data = [[x for i, x in enumerate(sol[:, 3]) if tree_size == sol[:, 2][i]] for tree_size in tree_sizes]
    data = [[0] if x == [] else x for x in data]
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    plt.xticks(range(1, len(tree_sizes) + 1), tree_sizes)
    plt.xlabel("Tree size")
    plt.ylabel("Success rate")
    plt.title("Mountain Car")
    plt.legend()
    plt.show()

def cartpole_solutions():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define the arrays
    # A = [5, 5, 5, 7, 9, 13, 13]
    # B = [0.1, 0.1, 0.3, 0.1, 0.2, 0.4, 0.2]
    # C = [1, 1, 1, 2, 2, 2, 2]

    success_rates = []
    tree_sizes = []
    groups = []

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL_p002_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-RP_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/cartpole_CRO__2023_03_23-15_22_26_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-CRO__2023_02_10-11_44_21_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-RP-CRO_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    # Create a dictionary to map A values to B values
    data_dict = {'Tree Size': np.int_(tree_sizes), 'Success Rate': success_rates, 'Group': groups}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(14, 10))

    # Group the DataFrame by A values and plot Swarmplots for each group
    # groups = df.groupby('Tree Size')
    sns.swarmplot(x='Tree Size', y='Success Rate', data=df, ax=ax, hue='Group')

    # Set the x-axis label
    ax.set_xlabel('Tree Size')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4, 0, 3, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Algorithm')

    # Show the plot
    plt.show()

def mountain_car_solutions():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define the arrays
    # A = [5, 5, 5, 7, 9, 13, 13]
    # B = [0.1, 0.1, 0.3, 0.1, 0.2, 0.4, 0.2]
    # C = [1, 1, 1, 2, 2, 2, 2]

    success_rates = []
    tree_sizes = []
    groups = []

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL_p015_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_CRO__2023_02_08-11_08_34_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-CRO__2023_03_21-13_15_47_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP-CRO__2023_03_24-12_15_07_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    # Create a dictionary to map A values to B values
    data_dict = {'Tree Size': np.int_(tree_sizes), 'Success Rate': success_rates, 'Group': groups}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(14, 10))

    # Group the DataFrame by A values and plot Swarmplots for each group
    # groups = df.groupby('Tree Size')
    sns.swarmplot(x='Tree Size', y='Success Rate', data=df, ax=ax, hue='Group')

    # Set the x-axis label
    ax.set_xlabel('Tree Size')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4, 0, 3, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Algorithm')

    # Show the plot
    plt.show()

def lunar_lander_solutions():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define the arrays
    # A = [5, 5, 5, 7, 9, 13, 13]
    # B = [0.1, 0.1, 0.3, 0.1, 0.2, 0.4, 0.2]
    # C = [1, 1, 1, 2, 2, 2, 2]

    success_rates = []
    tree_sizes = []
    groups = []

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL_a0001_p0004_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-RP_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_CRO_2023_02_27-18_53_54_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-CRO__2023_03_02-11_35_43_reevaluated_2.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    sol = parse_file("../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-RP-CRO_2023_02_17-12_56_21_reevaluated.txt")
    success_rates.extend(sol[:, 3])
    tree_sizes.extend(sol[:, 2])
    groups.extend(['IL $\\rightarrow$ RP $\\rightarrow$ CRO-DT-RL'] * len(sol[:, 3]))

    # Create a dictionary to map A values to B values
    data_dict = {'Tree Size': np.int_(tree_sizes), 'Success Rate': success_rates, 'Group': groups}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(14, 10))

    # Group the DataFrame by A values and plot Swarmplots for each group
    # groups = df.groupby('Tree Size')
    # sns.swarmplot(x='Tree Size', y='Success Rate', data=df, ax=ax, hue='Group')
    sns.scatterplot(x='Tree Size', y='Success Rate', data=df, ax=ax, hue='Group')

    # Set the x-axis label
    ax.set_xlabel('Tree Size')

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [4, 0, 3, 1, 2]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Algorithm')

    # Show the plot
    plt.show()

def generic_plot(files, algos):
    success_rates = []
    tree_sizes = []
    groups = []

    for i, file in enumerate(files):
        sol = parse_file(file)
        success_rates.extend(sol[:, 3])
        tree_sizes.extend(sol[:, 2])
        groups.extend([algos[i]] * len(sol[:, 3]))

    data_dict = {'Tree Size': np.int_(tree_sizes), 'Success Rate': success_rates, 'Group': groups}
    df = pd.DataFrame(data_dict)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.swarmplot(x='Tree Size', y='Success Rate', data=df, ax=ax, hue='Group')
    ax.set_xlabel('Tree Size')
    plt.show()

if __name__ == '__main__':
    # cartpole_solutions()
    # mountain_car_solutions()
    # lunar_lander_solutions()

    generic_plot([
            # "../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL_p015_reevaluated.txt",
            # "../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP_reevaluated.txt",
            "../../../CRO_DT_RL/results/complete/mountain-car_CRO__2023_02_08-11_08_34_reevaluated.txt",
            "../../../CRO_DT_RL/results/complete/mountain-car_IL-CRO__2023_03_21-13_15_47_reevaluated.txt",
            "../../../CRO_DT_RL/results/complete/mountain-car_IL-RP-CRO__2023_03_24-12_15_07_reevaluated.txt"
        ],
        [
            "CRO-DT-RL (R)",
            "CRO-DT-RL (IL)",
            "CRP-DT-RL (P)"
        ])