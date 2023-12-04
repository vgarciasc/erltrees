import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import pandas as pd

def parse_file(filename):
    solutions = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("Tree #"):
                tree_strs = line.split(" ")
                solutions.append((float(tree_strs[3]), float(tree_strs[5][:-1]), float(tree_strs[7][:-1]), float(tree_strs[10][:-2])))
    return np.array(solutions)

if __name__ == "__main__":
    CRO_R = ([
                 # ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_r\\-110\mountain-car_CRO-R__2023_10_27-15_12_23_reevaluated.txt",
                 ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\mountain-car_CRO-R.txt",
                  "Fitness A"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_r\\-110\mountain-car_CRO-R_no-sigma_2023_10_28-18_40_53_reevaluated.txt",
               "Fitness B"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_r\\-110\mountain-car_CRO-R_no-alpha_2023_10_28-07_55_02_reevaluated.txt",
               "Fitness C"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_r\\-110\mountain-car_CRO-R_no-alpha_no-sigma_2023_10_29-04_26_48_reevaluated.txt",
               "Fitness D")], "MENS-DT-RL (R)")

    CRO_IL = ([
                  # ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_il\\mountain-car_CRO-IL_2023_10_30-15_50_10_reevaluated.txt",
                  ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\mountain-car_CRO-IL.txt",
               "Fitness A"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_il\\mountain-car_CRO-IL_no-sigma_2023_10_31-11_47_08.txt",
               "Fitness B"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_il\\mountain-car_CRO-IL_no-alpha_2023_10_31-01_49_37.txt",
               "Fitness C"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_il\\mountain-car_CRO-IL_no-alpha_no-sigma_2023_11_07-15_23_37.txt",
               "Fitness D")], "MENS-DT-RL (IL)")

    CRO_P = ([
                  # ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_p\\mountain-car_CRO-P_2023_11_01-15_07_44.txt",
                 ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\mountain-car_CRO-P.txt",
                  "Fitness A"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_p\\mountain-car_CRO-P_no-sigma_2023_11_02-12_59_27.txt",
               "Fitness B"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_p\\mountain-car_CRO-P_no-alpha_2023_11_02-04_07_21.txt",
               "Fitness C"),
              ("C:\\Users\\ViníciusGarcia\\Documents\\Github\\CRO_DT_RL\\results\\ablation\cro_p\\mountain-car_CRO-P_no-alpha_no-sigma_2023_11_08-04_52_50.txt",
               "Fitness D")], "MENS-DT-RL (P)")

    for config_files, config_name in [CRO_R, CRO_IL, CRO_P]:
        avg_rewards = []
        std_rewards = []
        tree_sizes = []
        success_rates = []
        groups = []

        for (file, group) in config_files:
            sol = parse_file(file)
            avg_rewards.extend(sol[:, 0])
            std_rewards.extend(sol[:, 1])
            tree_sizes.extend(sol[:, 2])
            success_rates.extend(sol[:, 3])
            groups.extend([group] * len(sol[:, 0]))

        df = pd.DataFrame({
            "Average reward": avg_rewards,
            "Standard deviation of reward": std_rewards,
            "Tree size": tree_sizes,
            "Success rate": success_rates,
            "Model": groups
        })

        sns.set_palette("deep")

        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        sns.boxplot(data=df, x="Model", y="Average reward", hue="Model", ax=axs[0, 0])
        sns.boxplot(data=df, x="Model", y="Standard deviation of reward", hue="Model", ax=axs[0, 1])
        sns.boxplot(data=df, x="Model", y="Tree size", hue="Model", ax=axs[1, 0])
        sns.boxplot(data=df, x="Model", y="Success rate", hue="Model", ax=axs[1, 1])
        for ax in axs.flat:
            ax.set(xlabel=None)

        plt.suptitle(f"Ablation study - Mountain Car - {config_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.931)
        plt.savefig(f"ablation_{config_name}.pdf", dpi=200)