import pdb
import json
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import argparse
from matplotlib.colors import ListedColormap
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.rl.utils import fill_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car Plotter')
    parser.add_argument('-f','--filename', help="Input file", required=True, type=str)
    args = vars(parser.parse_args())

    config = get_config("mountain_car")
    with open("models/dagger_mc_population_2.txt", "r") as f:
        tree_strs = json.load(f)
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]

    with open(args['filename'], "rb") as f:
        X, y = pickle.load(f)

    X_left = np.array([X[j] for j, y_j in enumerate(y) if y_j == 0])
    X_nop = np.array([X[j] for j, y_j in enumerate(y) if y_j == 1])
    X_right = np.array([X[j] for j, y_j in enumerate(y) if y_j == 2])
    
    for i, tree in enumerate(trees):
        print(f"Total: {i} / {len(trees)}")
        plt.figure()
        plt.scatter(X_left[:,0], X_left[:,1], color='orange', s=4, label='Move Left')
        plt.scatter(X_nop[:,0], X_nop[:,1], color='grey', s=4, label='Do nothing')
        plt.scatter(X_right[:,0], X_right[:,1], color='blue', s=4, label='Move Right')

        h = .005  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = np.array([tree.act(x) for x in np.c_[xx.ravel(), yy.ravel()]])

        colors = ('orange', 'blue')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)
        plt.xlim((-1.3, 0.7))
        plt.ylim((-0.1, 0.1))
        plt.xlabel("Car Position")
        plt.ylabel("Car Velocity")
        plt.legend()

        fill_metrics(config, [tree], alpha=0.0, episodes=100, should_norm_state=False, penalize_std=False, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)
        plt.title(f"Tree #{i}\nReward: {'{:.3f}'.format(tree.reward)} +- {'{:.3f}'.format(tree.std_reward)}, size: {tree.get_tree_size()}, SR: {'{:.2f}'.format(tree.success_rate)}")
        plt.savefig(f"mountaincar_plot_{i}.png")
        
        plt.clf()
        # plt.show()
    
    print(f"average reward: {np.mean([t.reward for t in trees])} +- {np.mean([t.std_reward for t in trees])}, SR: {np.mean([t.success_rate for t in trees])}, size: {np.mean([t.get_tree_size() for t in trees])}")
    pdb.set_trace()