import numpy as np
from rich.console import Console

console = Console()

def printv(str, verbose=False):
    if verbose:
        console.log(str)

def save_history_to_file(config, history, filepath, prefix=""):
    string = prefix
    
    if history is not None:
        trees, rewards, sizes, _ = zip(*history)
        successes = [1 if r > config["reward_to_success"] else 0 for r in rewards]
        trees = np.array(trees)

        string += f"Mean Best Reward: {np.mean(rewards)} +- {np.std(rewards)}\n"
        string += f"Mean Best Size: {np.mean(sizes)}\n"
        string += f"Average Evaluations to Success: -------\n"
        string += f"Success Rate: {np.mean(successes)}\n"
        string += "\n-----\n\n"

        for i, tree in enumerate(trees):
            string += f"Tree #{i} (Reward: {tree.reward}, Size: {tree.get_tree_size()})\n"
            string += "----------\n"
            string += str(tree)
            string += "\n"
    
    with open(filepath, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def get_trees_from_logfile(filepath):
    tree_strings = []
    with open(filepath) as f:
        curr_line_idx = 0
        lines = f.readlines()

        while curr_line_idx < len(lines):
            if lines[curr_line_idx].strip() == "----------":
                curr_line_idx += 1
                start_line = curr_line_idx
                while lines[curr_line_idx] != "\n":
                    curr_line_idx += 1
                end_line = curr_line_idx
                tree_string = "\n" + "".join(lines[start_line:end_line]).rstrip()
                tree_strings.append(tree_string)
            else:
                curr_line_idx += 1
    return tree_strings