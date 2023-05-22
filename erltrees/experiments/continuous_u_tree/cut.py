from erltrees.experiments.continuous_u_tree.qtree import QTree
from erltrees.rl.configs import get_config
from erltrees.rl.utils import collect_rewards
import numpy as np

if __name__ == "__main__":
    config = get_config("cartpole")
    tree = QTree(config)

    data_gathering_phase_episodes = 1000
    config['alpha'] = 0.01
    config['gamma'] = 0.99
    config['splitting_threshold'] = 0.0001

    for round in range(10):
        print(f"Round {round}.")
        for leaf in tree.get_all_leaves():
            leaf.transition_history = []

        # Data gathering phase
        for episode in range(data_gathering_phase_episodes):
            env = config["maker"]()
            state = env.reset()
            done = False

            while not done:
                action = tree.act(state)
                next_state, reward, done, _ = env.step(action)
                tree.get_leaf(state).record_transition(state, action, next_state, 0 if done else reward, done)
                state = next_state

            env.close()

        # Processing phase
        for leaf in tree.get_all_leaves():
            leaf.datapoints = []
            for state, action, next_state, reward, done in leaf.transition_history:
                q_value = reward + config['gamma'] * np.max(leaf.q_values)
                leaf.datapoints.append(((state, action), q_value))

            # Splitting phase
            best_split = None
            best_variance = None
            original_variance = np.var([q_value for (_, q_value) in leaf.datapoints])

            for i, (attribute, _, _) in enumerate(config['attributes']):
                splitting_data = [(state[i], q_value) for ((state, action), q_value) in leaf.datapoints]
                splitting_data.sort(key=lambda x: x[0])
                splitting_data = np.array(splitting_data)

                for split in range(splitting_data.shape[0] - 1):
                    left_attribs, left_q_values = splitting_data[:split + 1].T
                    right_attribs, right_q_values = splitting_data[split + 1:].T
                    split_value = (left_attribs[-1] + right_attribs[0]) / 2

                    variance = (np.var(left_q_values) * left_attribs.shape[0] +
                                np.var(right_q_values) * right_attribs.shape[0]) \
                               / splitting_data.shape[0]

                    if best_variance is None or variance < best_variance:
                        if variance - original_variance > config['splitting_threshold']:
                            best_split = (i, split_value)
                            best_variance = variance

            if best_split is not None:
                leaf.split(*best_split)
                print(f"Splitting on attribute {config['attributes'][best_split[0]][0]} with value {best_split[1]}. "
                      f"New tree size: {tree.get_tree_size()}")

        print(f"Running Q-learning on {tree.get_tree_size()} leaves.")
        tree.q_learn(episodes=20000, verbose=False)
        r = collect_rewards(config, tree, 100, False)
        print(f"Average reward: {np.mean(r)}")
