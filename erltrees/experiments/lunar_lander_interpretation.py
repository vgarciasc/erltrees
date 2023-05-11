import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from rich import print

total_visits = 1
def print_ll(tree, state):
    global total_visits
    stack = [(tree, 1, True)]
    output = ""

    while len(stack) > 0:
        node, depth, selected = stack.pop()
        output += "-" * depth + " "

        if selected:
            output += "[green]"
        else:
            output += "[red]"

        if node.is_leaf():
            output += (tree.config['actions'][node.label]).upper()
            output += f" ({(node.visits / total_visits):.5f})"
        else:
            output += tree.config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.5f}'.format(node.threshold)

            if node.right:
                stack.append((node.right, depth + 1, selected and state[node.attribute] > node.threshold))
            if node.left:
                stack.append((node.left, depth + 1, selected and state[node.attribute] <= node.threshold))

        if selected:
            output += "[/green]"
        else:
            output += "[/red]"
        output += "\n"

    print(output)

if __name__ == "__main__":
    tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"
    # tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Leg 2 is Touching <= 0.50000\n--- Y Velocity <= -0.09085\n---- Angle <= -0.04364\n----- Y Velocity <= -0.25810\n------ Y Position <= 0.20415\n------- MAIN ENGINE\n------- Angular Velocity <= -0.18925\n-------- LEFT ENGINE\n-------- X Velocity <= -0.17510\n--------- MAIN ENGINE\n--------- Angular Velocity <= -0.02175\n---------- LEFT ENGINE\n---------- MAIN ENGINE\n------ X Velocity <= 0.02710\n------- RIGHT ENGINE\n------- LEFT ENGINE\n----- Y Velocity <= -0.28725\n------ X Velocity <= -0.39615\n------- RIGHT ENGINE\n------- MAIN ENGINE\n------ Angle <= 0.21595\n------- X Position <= -0.02269\n-------- RIGHT ENGINE\n-------- Angular Velocity <= 0.18515\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n------- RIGHT ENGINE\n---- Y Position <= 0.00074\n----- NOP\n----- Angle <= 0.02441\n------ LEFT ENGINE\n------ RIGHT ENGINE\n--- Y Velocity <= -0.06200\n---- MAIN ENGINE\n---- Angle <= -0.21080\n----- LEFT ENGINE\n----- NOP\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"

    config = get_config("lunar_lander")
    tree = Individual.read_from_string(config, tree_str)
    env = config['maker']()

    history = []
    total_rewards = []

    for episode in range(1000):
        done = False
        state = env.reset()
        total_rewards.append(0)

        while not done:
            action = tree.act(state)
            state, reward, done, _ = env.step(action)

            env.render()
            # time.sleep(0.1)
            # print_ll(tree, state)

            # print(f"X Position: [red]{'{:.3f}'.format(state[0])}[/red], Y Position: [red]{'{:.3f}'.format(state[1])}[/red],\n "
            #       f"X Velocity: [red]{'{:.3f}'.format(state[2])}[/red], Y Velocity: [red]{'{:.3f}'.format(state[3])}[/red],\n "
            #       f"Angle: [red]{'{:.3f}'.format(state[4])}[/red], Angular Velocity: [red]{'{:.3f}'.format(state[5])}[/red],\n "
            #       f"Leg 1 is Touching: [red]{'{:.3f}'.format(state[6])}[/red], Leg 2 is Touching: [red]{'{:.3f}'.format(state[7])}[/red]")

            total_rewards[-1] += reward
            history.append((state, action))

            total_visits += 1

        print(f"Episode {episode + 1} finished with reward {total_rewards[-1]}")

    print(f"Average reward: {np.mean(total_rewards)}")
    print(f"Average success rate: {np.mean([1 if r > 200 else 0 for r in total_rewards])}")

    states, actions = zip(*history)
    states = np.array(states)
    actions = np.array(actions)

    plt.subplots(1, 3, figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(states[:, 0], bins=30, density=True, alpha=0.5, color="blue")
    plt.xlabel("X Position")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(states[:, 4], bins=30, density=True, alpha=0.5, color="red")
    plt.xlabel("Angle")

    plt.subplot(1, 3, 3)
    plt.hist(states[:, 5], bins=30, density=True, alpha=0.5, color="green")
    plt.xlabel("Angular Velocity")

    plt.show()

    print(f"Total visits: {total_visits}")
    print_ll(tree, state)

    print(f"Percentage of angles > 0 : {len([s for s in states[:, 4] if s > 0]) / len(states[:, 4])}")
    print(f"Percentage of angles < 0 : {len([s for s in states[:, 4] if s < 0]) / len(states[:, 4])}")
