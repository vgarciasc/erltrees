import numpy as np
import pdb
import json
import matplotlib.pyplot as plt

def load_file_json(filepath):
    with open(filepath, "r") as f:
        content = json.load(f)
    return content

def clean_json(content):
    avgs, stds, sizes = [], [], []
    for avg, std, size in zip(content["avg_rewards"], content["deviations"], content["sizes"]):
        try:
            idx = sizes.index(size)
        except:
            idx = -1
        
        if idx == -1:
            avgs.append(avg)
            stds.append(std)
            sizes.append(size)
        elif avgs[idx] - stds[idx] < avg - std:
            avgs[idx] = avg
            stds[idx] = std
    
    avgs = [x for _, x in sorted(zip(sizes, avgs), reverse=True)]
    stds = [x for _, x in sorted(zip(sizes, stds), reverse=True)]
    sizes.sort()

    return avgs, stds, sizes

if __name__ == "__main__":
    filepath_bc = "data/imitation_learning/behavioral_cloning_grid_2022_11_10-12_01_31.txt"
    bc_content = load_file_json(filepath_bc)
    filepath_dg = "data/imitation_learning/dagger_grid_2022_11_10-06_19_30.txt"
    dg_content = load_file_json(filepath_dg)

    avgs, stds, x = clean_json(bc_content)
    plt.plot(x, avgs, color="red", label="Behavioral Cloning")
    plt.fill_between(x, 
        np.array(avgs) - np.array(stds),
        np.array(avgs) + np.array(stds),
        color="red", alpha=0.2)

    avgs, stds, x = clean_json(dg_content)
    plt.plot(x, avgs, color="blue", label="Dagger")
    plt.fill_between(x, 
        np.array(avgs) - np.array(stds),
        np.array(avgs) + np.array(stds),
        color="blue", alpha=0.2)

    plt.xlabel("Tree size")
    plt.ylabel("Reward")
    # plt.gca().invert_xaxis()
    plt.legend()
    plt.show()