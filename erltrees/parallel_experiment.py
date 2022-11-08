import json
from math import sqrt
import pdb
from joblib import Parallel, delayed
from time import time
import numpy as np
from rich import print

from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl
from rich.console import Console

console = Console()

alpha = 0.5
episodes = 30
num_trees = 100

# Initialize population

# tree_str = "\n- Car Velocity <= -0.001\n-- LEFT\n-- Car Position <= -0.096\n--- Car Position <= 2.000\n---- RIGHT\n---- RIGHT\n--- Car Velocity <= 0.027\n---- RIGHT\n---- NOP"
# tree_str = "\n- Car Velocity <= -0.02401\n-- LEFT\n-- Car Velocity <= 0.20463\n--- Car Position <= -0.17120\n---- RIGHT\n---- LEFT\n--- RIGHT"
# tree_str = "\n- Car Velocity <= -0.04286\n-- LEFT\n-- Car Velocity <= 0.07000\n--- Car Velocity <= 0.46567\n---- Car Position <= -0.30667\n----- RIGHT\n----- Car Position <= -0.21000\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"
# config = get_config("mountain_car")

tree_str = "\n- Leg 1 is Touching <= 0.50000\n-- Angle <= -0.01814\n--- Y Velocity <= -0.04140\n---- Angular Velocity <= -0.03520\n----- LEFT ENGINE\n----- X Velocity <= -0.01060\n------ MAIN ENGINE\n------ Angular Velocity <= 0.02240\n------- LEFT ENGINE\n------- X Velocity <= 0.05960\n-------- MAIN ENGINE\n-------- LEFT ENGINE\n---- NOP\n--- Y Velocity <= 0.25919\n---- X Velocity <= -0.01300\n----- Y Velocity <= -0.05960\n------ Leg 1 is Touching <= -0.16376\n------- MAIN ENGINE\n------- NOP\n------ X Velocity <= -0.02680\n------- RIGHT ENGINE\n------- RIGHT ENGINE\n----- Y Position <= 0.16467\n------ Y Position <= 0.72484\n------- Y Position <= 1.03512\n-------- X Velocity <= 0.79464\n--------- Y Velocity <= -0.02780\n---------- MAIN ENGINE\n---------- NOP\n--------- MAIN ENGINE\n-------- MAIN ENGINE\n------- RIGHT ENGINE\n------ Y Velocity <= -0.04700\n------- Angle <= 0.07257\n-------- X Position <= -0.01733\n--------- Angular Velocity <= 0.01700\n---------- X Position <= 0.00000\n----------- MAIN ENGINE\n----------- X Position <= -0.91163\n------------ NOP\n------------ MAIN ENGINE\n---------- RIGHT ENGINE\n--------- Angle <= 0.01464\n---------- MAIN ENGINE\n---------- MAIN ENGINE\n-------- Angular Velocity <= -0.02160\n--------- MAIN ENGINE\n--------- RIGHT ENGINE\n------- Angle <= 0.05061\n-------- LEFT ENGINE\n-------- RIGHT ENGINE\n---- RIGHT ENGINE\n-- Angle <= 0.46023\n--- Y Velocity <= -0.01360\n---- MAIN ENGINE\n---- NOP\n--- RIGHT ENGINE"
config = get_config("lunar_lander")

trees = [Individual.read_from_string(config, string=tree_str) for _ in range(num_trees)]
norm_state = True
print(f"Population has {len(trees)} trees.")

# for n_jobs in [8, 16, 32]:
for n_jobs in [-1, 2, 4, 8, 16, 32]:
    title = f"PARALLEL w/ {n_jobs} JOBS" if n_jobs > 1 else "SEQUENTIAL"
    console.rule(title)
    TIME_START = time()
    rl.fill_metrics(config=config, trees=trees,
        alpha=alpha, episodes=episodes, 
        should_norm_state=norm_state,
        n_jobs=n_jobs)
    TIME_END = time()
    print(f"Elapsed time: {TIME_END - TIME_START} seconds")
    rewards = [t.reward for t in trees]
    print(f"Average reward: {np.mean(rewards)} +- {np.std(rewards)}")

pdb.set_trace()