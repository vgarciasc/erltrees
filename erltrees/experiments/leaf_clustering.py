import numpy as np

from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.viz.graph_utils import convert_tree_to_tree_string, convert_tree_string_to_dot, \
    convert_tree_to_tree_string_with_communities, COLORS

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import graphviz
from collections import Counter

if __name__ == "__main__":
    config = get_config("lunar_lander")

    tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"
    agent = Individual.read_from_string(config, tree_str)

    leaves = agent.get_node_list(get_leaves=True)
    cooccurrences = Counter()

    leaf_history = []

    env = config["maker"]()
    for episode in range(1000):
        done = False
        state = env.reset()
        curr_leaf = agent.get_leaf(state)
        leaf_history.append([])

        while not done:
            # env.render()
            action = agent.act(state)
            state, reward, done, info = env.step(action)

            next_leaf = agent.get_leaf(state)
            cooccurrences[(curr_leaf, next_leaf)] += 1
            leaf_history[-1].append(curr_leaf)

            curr_leaf = next_leaf

    env.close()

    for (leaf1, leaf2), count in cooccurrences.items():
        print(f"{leaves.index(leaf1)} -> {leaves.index(leaf2)}: {count}")

    # create co-occurrence network
    G = nx.Graph()
    for (leaf1, leaf2), count in cooccurrences.items():
        G.add_edge(f"{leaves.index(leaf1)}",
                   f"{leaves.index(leaf2)}",
                   weight=count)

    # run louvain community detection
    louvain_communities = nx.algorithms.community.louvain_communities(G, weight="weight")
    greedy_modularity_communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    communities = louvain_communities

    tree_string = convert_tree_to_tree_string_with_communities(config, agent, louvain_communities)
    dot_string = convert_tree_string_to_dot(tree_string)
    graph = graphviz.Source(dot_string, format="png")
    graph.render(filename="tmp_louvain", cleanup=True)
    graph.view()

    # tree_string = convert_tree_to_tree_string_with_communities(config, agent, greedy_modularity_communities)
    # dot_string = convert_tree_string_to_dot(tree_string)
    # graph = graphviz.Source(dot_string, format="png")
    # graph.render(filename="tmp_greedy", cleanup=True)
    # graph.view()

    get_community = lambda leaf_id : [i for (i, c) in enumerate(communities) if str(leaf_id) in c]

    # plot last leaf history
    plt.subplots(2, 1, figsize=(12, 5))
    node_ids = [[leaves.index(leaf) for leaf in history] for history in leaf_history]
    comm_ids = [[get_community(leaf_id)[0] if len(get_community(leaf_id)) > 0 else -1 for leaf_id in history] for history in node_ids]
    x = [[i for i in range(len(history))] for history in node_ids]
    x_percentage = [[i / len(history) for i in range(len(history))] for history in node_ids]
    episode_lengths = [len(history) for history in leaf_history]
    max_episode_length = max(episode_lengths)
    ax = plt.subplot(2, 1, 1)
    ax.set_xlabel("Episode Progress")
    ax.set_ylabel("Leaf ID")
    sns.scatterplot(x=x_percentage[-1],
                    y=node_ids[-1],
                    hue=comm_ids[-1],
                    palette=COLORS,
                    edgecolor='none', alpha=0.5,
                    legend=False, ax=ax)
    ax = plt.subplot(2, 1, 2)
    ax.set_xlabel("Episode Progress")
    ax.set_ylabel("Leaf group")
    sns.scatterplot(x=x_percentage[-1],
                    y=comm_ids[-1],
                    hue=comm_ids[-1],
                    palette=COLORS,
                    edgecolor='none', alpha=0.5,
                    legend=False, ax=ax)
    plt.suptitle("Last Episode", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

    # plot last leaf history
    plt.subplots(2, 1, figsize=(12, 5))
    node_ids = [[leaves.index(leaf) for leaf in history] for history in leaf_history]
    comm_ids = [[get_community(leaf_id)[0] if len(get_community(leaf_id)) > 0 else -1 for leaf_id in history] for history in node_ids]
    x = [[i for i in range(len(history))] for history in node_ids]
    x_percentage = [[i / len(history) for i in range(len(history))] for history in node_ids]
    episode_lengths = [len(history) for history in leaf_history]
    max_episode_length = max(episode_lengths)
    ax = plt.subplot(2, 1, 1)
    ax.set_xlabel("Episode Progress")
    ax.set_ylabel("Leaf ID")
    sns.scatterplot(x=np.concatenate(x_percentage),
                    y=np.concatenate(node_ids),
                    hue=np.concatenate(comm_ids),
                    palette=COLORS,
                    edgecolor='none', alpha=0.1,
                    legend=False, ax=ax)
    ax = plt.subplot(2, 1, 2)
    ax.set_xlabel("Episode Progress")
    ax.set_ylabel("Leaf group")
    sns.scatterplot(x=np.concatenate(x_percentage),
                    y=np.concatenate(comm_ids),
                    hue=np.concatenate(comm_ids),
                    palette=COLORS,
                    edgecolor='none', alpha=0.1,
                    legend=False, ax=ax)
    plt.suptitle("All Episodes", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

