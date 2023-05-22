import matplotlib.pyplot as plt
import numpy as np
from erltrees.rl.configs import get_config
from rich import print


class QTree:
    def __init__(self, config=None, attribute=None, threshold=None, left=None, right=None, q_values=None):
        self.config = config
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right

        # self.q_values = q_values if q_values is not None else np.zeros(config["n_actions"])
        self.q_values = q_values if q_values is not None else np.random.uniform(-1, 1, config["n_actions"])
        self.transition_history = []
        self.datapoints = []

    def act(self, state):
        return np.argmax(self.get_leaf(state).q_values)

    def get_leaf(self, state):
        if self.is_leaf():
            return self
        if state[self.attribute] <= self.threshold:
            return self.left.get_leaf(state)
        else:
            return self.right.get_leaf(state)

    def is_leaf(self):
        return self.left is None and self.right is None

    def record_transition(self, state, action, next_state, reward, done):
        self.transition_history.append((state, action, next_state, reward, done))

    def update_datapoint_values(self):
        self.datapoints = []
        for state, action, next_state, reward, done in self.transition_history:
            next_leaf = self.get_leaf(next_state)
            next_q = 0 if done else np.max(next_leaf.q_values)
            self.datapoints.append(reward + config['gamma'] * next_q)

    def get_all_leaves(self):
        if self.is_leaf():
            return [self]
        else:
            return self.left.get_all_leaves() + self.right.get_all_leaves()

    def split(self, attribute, threshold):
        self.attribute = attribute
        self.threshold = threshold

        self.left = QTree(self.config)
        self.right = QTree(self.config)

        for ((state, action), q_value) in self.datapoints:
            if state[attribute] <= threshold:
                self.left.datapoints.append(((state, action), q_value))
            else:
                self.right.datapoints.append(((state, action), q_value))

        self.datapoints = []

    def get_tree_size(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + self.left.get_tree_size() + self.right.get_tree_size()

    def update_q_value(self, state, action, reward, next_state, done):
        leaf = self.get_leaf(state)
        next_leaf = self.get_leaf(next_state)

        next_q = 0 if done else max(*next_leaf.q_values)
        delta_q = self.config['alpha'] * (reward + self.config['gamma'] * next_q - leaf.q_values[action])
        leaf.q_values[action] += delta_q

    def q_learn(self, episodes, verbose=False):
        total_rewards = []

        for episode in range(episodes):
            env = self.config["maker"]()
            state = env.reset()
            done = False
            total_rewards.append(0)

            while not done:
                if np.random.uniform() < max(0.5, (1 - episode / episodes)) and episode < episodes * 0.9:
                    action = np.random.randint(0, self.config["n_actions"])
                else:
                    action = self.act(state)

                next_state, reward, done, _ = env.step(action)

                leaf = self.get_leaf(state)
                leaf.update_q_value(state, action, reward, next_state, done)

                total_rewards[-1] += reward
                state = next_state
            env.close()

            if episode % 100 == 0 and verbose:
                print(f"Episode {episode} || Total reward: {total_rewards[-1]}")
                # print(f"Episode {episode} || Total reward: {total_rewards[-1]} || "
                      # f"Actions: {self.config['actions'][np.argmax(self.left.left.q_values)]}, "
                      # f"{self.config['actions'][np.argmax(self.left.right.q_values)]}, "
                      # f"{self.config['actions'][np.argmax(self.right.q_values)]} || "
                      # f"Q-values: [{', '.join(['{:.3f}'.format(s) for s in self.left.left.q_values])}], "
                      # f"[{', '.join(['{:.3f}'.format(s) for s in self.left.right.q_values])}], "
                      # f"[{', '.join(['{:.3f}'.format(s) for s in self.right.q_values])}]")

        return total_rewards


def plot_rewards(total_rewards, title):
    total_rewards = np.array(total_rewards)
    plt.plot(np.mean(total_rewards, axis=0), color="blue")
    plt.fill_between(range(total_rewards.shape[1]),
                     np.mean(total_rewards, axis=0) - np.std(total_rewards, axis=0),
                     np.mean(total_rewards, axis=0) + np.std(total_rewards, axis=0),
                     color="blue", alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    tree_string = "- Pole Angular Velocity <= 0.074\n-- Pole Angle <= 0.022\n--- LEFT\n--- RIGHT\n-- RIGHT"
    config = get_config("cartpole")
    config['alpha'] = 0.001
    config['gamma'] = 1

    rs = []
    for simulation in range(10):
        tree = QTree(config, attribute=3, threshold=0.074,
                     left=QTree(config, attribute=2, threshold=0.022,
                                left=QTree(config),
                                right=QTree(config)),
                     right=QTree(config))
        r = tree.q_learn(50000, verbose=True)
        rs.append(r)
    plot_rewards(rs, "Continuous U-Tree")
