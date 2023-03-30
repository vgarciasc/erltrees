import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import pandas as pd
from cycler import cycler

ALPHA = 1


def parse_file(filename):
    solutions = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("Tree #"):
                tree_strs = line.split(" ")
                avg_reward = float(tree_strs[3])
                std_reward = float(tree_strs[5][:-1])
                tree_size = float(tree_strs[7][:-1])
                success_rate = float(tree_strs[10][:-2])
                fitness = avg_reward - std_reward - tree_size * ALPHA

                solutions.append((fitness, avg_reward, std_reward, tree_size, success_rate))
    return np.array(solutions)


def boxplot_solutions(title, files, algos):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fitnesses, avg_rewards, std_rewards, success_rates, tree_sizes, groups = [], [], [], [], [], []

    for i in range(len(files)):
        sol = parse_file(files[i])
        fitnesses.extend(sol[:, 0])
        avg_rewards.extend(sol[:, 1])
        std_rewards.extend(sol[:, 2])
        success_rates.extend(sol[:, 3])
        tree_sizes.extend(sol[:, 4])
        groups.extend([algos[i]] * len(sol[:, 3]))

    # Create a dictionary to map A values to B values
    data_dict = {'Fitness': fitnesses, 'Average Reward': avg_rewards, 'Standard Deviation': std_rewards,
                 'Success Rate': success_rates, 'Tree Size': tree_sizes, 'Algorithm': groups}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group the DataFrame by A values and plot Swarmplots for each group
    # groups = df.groupby('Tree Size')
    sns.boxplot(df, x="Algorithm", y="Fitness", orient="v")

    # Show the plot
    plt.suptitle(title)
    plt.show()

if __name__ == '__main__':
    algorithm_names = [
                        # "IL",
                        # "IL $\\rightarrow$ RP",
                        "CRO-DT-RL (R)",
                        "CRO-DT-RL (IL)",
                        "CRO-DT-RL (P)"
    ]

    boxplot_solutions("Cartpole",
                      [
                          # "../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL_p002_reevaluated.txt",
                          #  "../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-RP_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/cartpole_CRO__2023_03_23-15_22_26_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-CRO__2023_02_10-11_44_21_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/cartpole_IL-RP-CRO_reevaluated.txt"
                      ],
                      algorithm_names)
    #
    boxplot_solutions("Mountain Car",
                      [
                          # "../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL_p015_reevaluated.txt",
                          #  "../../../CRO_DT_RL/results/complete/reevaluations/mountain-car_IL-RP_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/mountain-car_CRO__2023_02_08-11_08_34_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/mountain-car_IL-CRO__2023_03_21-13_15_47_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/mountain-car_IL-RP-CRO__2023_03_24-12_15_07_reevaluated.txt"
                      ],
                      algorithm_names)

    boxplot_solutions("Lunar Lander",
                      [
                          # "../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL_a0001_p0004_reevaluated.txt",
                          #  "../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-RP_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_CRO_2023_02_27-18_53_54_reevaluated.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-CRO__2023_03_02-11_35_43_reevaluated_2.txt",
                           "../../../CRO_DT_RL/results/complete/reevaluations/lunarlander_IL-RP-CRO_2023_02_17-12_56_21_reevaluated.txt"
                      ],
                      algorithm_names)

    # mountain_car_solutions()
    # lunar_lander_solutions()
