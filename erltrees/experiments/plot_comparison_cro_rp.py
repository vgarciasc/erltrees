import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read CSV
    df = pd.read_csv("../../data/reward_pruning_log.txt", sep=";", header=None,
                     names=['run_id', 'size', 'reward', 'std_reward', 'fitness', 'success_rate'],
                     index_col=False)
    df.reset_index(inplace=True, drop=True, names='index')

    # Select only data where first column is 3
    df = df[df['run_id'] == "2023-02-15_10-49-23"]

    x = df['size']
    y = df['reward']
    plt.plot(x, y, color="blue")
    plt.fill_between(x,
        df['reward'] - df['std_reward'],
        df['reward'] + df['std_reward'],
        color="blue", alpha=0.2)
    plt.gca().invert_xaxis()
    plt.show()

