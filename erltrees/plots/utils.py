import matplotlib.pyplot as plt
import numpy as np


def plot_full_log(config, log, filepath, display=False):
    plt.switch_backend('agg')

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    max_ngens = np.max([len(l) for l in log])
    for simul in log:
        for _ in range(len(simul), max_ngens):
            simul.append(simul[-1])
    x = range(max_ngens) # Number of generations

    fitnesses = np.array([[dt.fitness for _, dt in simul] for simul in log])
    tree_sizes = np.array([[dt.get_tree_size() for _, dt in simul] for simul in log])
    avg_rewards = np.array([[dt.reward for _, dt in simul] for simul in log])
    std_rewards = np.array([[dt.std_reward for _, dt in simul] for simul in log])

    ax = axs.flatten()[0]
    ax.plot(x, np.mean(fitnesses, axis=0), color="red")
    ax.fill_between(x, np.mean(fitnesses, axis=0) - np.std(fitnesses, axis=0),
                       np.mean(fitnesses, axis=0) + np.std(fitnesses, axis=0),
                    alpha=0.2, color="red")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average best fitness')

    ax = axs.flatten()[1]
    ax.plot(x, np.mean(avg_rewards, axis=0), color="blue")
    ax.fill_between(x, np.mean(avg_rewards, axis=0) - np.std(avg_rewards, axis=0),
                       np.mean(avg_rewards, axis=0) + np.std(avg_rewards, axis=0),
                    alpha=0.2, color="blue")
    ax.plot(x, np.ones(len(x)) * config["task_solution_threshold"],
            label="Solution threshold", color="black", linestyle="dashed")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average reward of best individual')

    ax = axs.flatten()[2]
    ax.plot(x, np.mean(std_rewards, axis=0), color="green")
    ax.fill_between(x, np.mean(std_rewards, axis=0) - np.std(std_rewards, axis=0),
                       np.mean(std_rewards, axis=0) + np.std(std_rewards, axis=0),
                    alpha=0.2, color="green")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average std. dev. of reward of best individual')

    ax = axs.flatten()[3]
    ax.plot(x, np.mean(tree_sizes, axis=0), color="orange")
    ax.fill_between(x, np.mean(tree_sizes, axis=0) - np.std(tree_sizes, axis=0),
                       np.mean(tree_sizes, axis=0) + np.std(tree_sizes, axis=0),
                    alpha=0.2, color="orange")
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average tree size of best individual')

    # Show the plot
    plt.suptitle(f"Full log of {filepath}")
    plt.tight_layout()
    plt.savefig(f"{filepath}.png")

    if display:
        plt.show()

    plt.close()