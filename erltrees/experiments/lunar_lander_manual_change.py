import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.rl.utils import collect_metrics
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
    # tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.25\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"
    # tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Leg 2 is Touching <= 0.50000\n--- Y Velocity <= -0.09085\n---- Angle <= -0.04364\n----- Y Velocity <= -0.25810\n------ Y Position <= 0.20415\n------- MAIN ENGINE\n------- Angular Velocity <= -0.18925\n-------- LEFT ENGINE\n-------- X Velocity <= -0.17510\n--------- MAIN ENGINE\n--------- Angular Velocity <= -0.02175\n---------- LEFT ENGINE\n---------- MAIN ENGINE\n------ X Velocity <= 0.02710\n------- RIGHT ENGINE\n------- LEFT ENGINE\n----- Y Velocity <= -0.28725\n------ X Velocity <= -0.39615\n------- RIGHT ENGINE\n------- MAIN ENGINE\n------ Angle <= 0.21595\n------- X Position <= -0.02269\n-------- RIGHT ENGINE\n-------- Angular Velocity <= 0.18515\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n------- RIGHT ENGINE\n---- Y Position <= 0.00074\n----- NOP\n----- Angle <= 0.02441\n------ LEFT ENGINE\n------ RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP"

    tree_strs = ["\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
        # "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
        # "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.25\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
                 "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= 0\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
        # "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= 0\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
        # "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= 0\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0.02441\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",
        "\n- Leg 1 is Touching <= 0.50000\n-- Y Velocity <= -0.09085\n--- Angle <= -0.04364\n---- Y Velocity <= -0.25810\n----- Y Position <= 0.20415\n------ MAIN ENGINE\n------ Angular Velocity <= -0.18925\n------- LEFT ENGINE\n------- X Velocity <= -0.17510\n-------- MAIN ENGINE\n-------- Angular Velocity <= -0.02175\n--------- LEFT ENGINE\n--------- MAIN ENGINE\n----- X Velocity <= 0.02710\n------ RIGHT ENGINE\n------ LEFT ENGINE\n---- Y Velocity <= -0.28725\n----- X Velocity <= -0.39615\n------ RIGHT ENGINE\n------ MAIN ENGINE\n----- Angle <= 0.21595\n------ X Position <= -0.02269\n------- RIGHT ENGINE\n------- Angular Velocity <= 0.18515\n-------- LEFT ENGINE\n-------- MAIN ENGINE\n------ RIGHT ENGINE\n--- Y Position <= 0.00074\n---- NOP\n---- Angle <= 0\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Y Velocity <= -0.06200\n--- MAIN ENGINE\n--- Angle <= -0.21080\n---- LEFT ENGINE\n---- NOP",]

    for i, tree_str in enumerate(tree_strs):
        config = get_config("lunar_lander")
        tree = Individual.read_from_string(config, tree_str)

        start_time = time.time()
        collect_metrics(config, [tree], 1.0, 100000, False, True, True, 200, True, False, 8)
        print(f"Time: {time.time() - start_time} seconds")

        print("----------")
        print(f"Tree {i}")
        print(f"Average reward: {tree.reward} +- {tree.std_reward}")
        print(f"Success rate: {tree.success_rate}")
        print("----------")