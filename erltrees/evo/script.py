import pdb
from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl
from erltrees.evo.evo_tree import Individual
import erltrees.evo.mutations as mutations
from rich import print

to_run = [
    # ("cartpole", "Silva et al. (2020)", "\n- Pole Angular Velocity <= 0.44\n-- Pole Angle <= 0.01\n--- LEFT\n--- RIGHT\n-- Pole Angular Velocity <= -0.3\n--- Pole Angle <= 0.0\n---- RIGHT\n---- LEFT\n--- Pole Angle <= -0.41\n---- LEFT\n---- RIGHT"),
    # ("cartpole", "Silva et al. (2020)'s best, as reported by Custode and Iacca", "\n- Pole Angular Velocity <= 0.18\n-- Pole Angle <= 0.00\n--- LEFT\n--- Pole Angular Velocity <= -0.3\n---- LEFT\n---- RIGHT\n-- Pole Angular Velocity <= -0.3\n--- Pole Angle <= 0.00\n---- RIGHT\n---- LEFT\n--- Pole Angle <= -0.41\n---- LEFT\n---- RIGHT"),
    # ("cartpole", "Custode and Iacca's best", "\n- Pole Angular Velocity <= 0.074\n-- Pole Angle <= 0.022\n--- LEFT\n--- RIGHT\n-- RIGHT"),
    # ("cartpole", "EL + IL best", ""),
    # ("cartpole", "RP best", ""),
    # ("mountain_car", "Custode and Iacca's best", "\n- Car Velocity <= -0.0001\n-- Car Position <= -0.9\n--- RIGHT\n--- LEFT\n-- Car Position <= -0.3\n--- Car Velocity <= 0.035\n---- Car Position <= -0.45\n----- RIGHT\n----- Car Position <= -0.4\n------ RIGHT\n------ LEFT\n---- RIGHT\n--- RIGHT"),
    # ("mountain_car", "EL + IL best", "\n- Car Velocity <= -0.00010\n-- Car Position <= -0.94656\n--- RIGHT\n--- LEFT\n-- Car Velocity <= 0.01796\n--- Car Position <= -0.38836\n---- RIGHT\n---- LEFT\n--- RIGHT"),
    # ("mountain_car", "RP best", "\n"),
    ("lunar_lander", "Silva et al. (2020)", "\n- Angular Velocity <= 0.04\n-- Y Velocity <= -0.35\n--- Angular Velocity <= -0.22\n---- Angle <= -0.04\n----- Leg 1 is Touching <= 0.5\n------ X Velocity <= 0.32\n------- Y Position <= -0.11\n-------- Angle <= 0.15\n--------- X Position <= -0.34\n---------- MAIN ENGINE\n---------- LEFT ENGINE\n--------- LEFT ENGINE\n-------- LEFT ENGINE\n------- MAIN ENGINE\n------ MAIN ENGINE\n----- LEFT ENGINE\n---- NOP\n--- NOP\n-- RIGHT ENGINE"),
]

if __name__ == "__main__":
    for (task_name, tree_name, tree_str) in to_run:
        config = get_config(task_name)
        tree = Individual.read_from_string(config, tree_str)

        rl.fill_metrics(config, [tree], alpha=1.0, episodes=10, should_norm_state=False, penalize_std=True, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)

        print(f"=== {task_name} => {tree_name}")
        print(f"Reward: {tree.reward} +- {tree.std_reward}")
        print(f"Num of leaves: {tree.get_tree_size()}")
        print(f"Success rate: {tree.success_rate}")

        print(tree)
    pdb.set_trace()