from erltrees.rl.configs import get_config
import erltrees.rl.utils as rl
from erltrees.evo.evo_tree import Individual

if __name__ == "__main__":
    tree_strs = ["\n- Pole Angle <= -0.04831\n-- LEFT\n-- Cart Velocity <= 0.55763\n--- RIGHT\n--- LEFT",
        "\n- Cart Velocity <= -0.29156\n-- RIGHT\n-- Pole Angle <= 0.04256\n--- LEFT\n--- RIGHT",
        "\n- Pole Angle <= -0.03456\n-- LEFT\n-- Pole Angular Velocity <= -0.24643\n--- LEFT\n--- RIGHT"]
    
    config = get_config("cartpole")
    trees = [Individual.read_from_string(config, tree_str) for tree_str in tree_strs]

    for i, tree in enumerate(trees):
        tree.denormalize_thresholds()
        print(f"Tree #{i}")
        print(tree)
        rl.collect_metrics(config, [tree], alpha=0.0, episodes=1000, should_norm_state=False, penalize_std=False, 
            should_fill_attributes=True, task_solution_threshold=config["task_solution_threshold"], n_jobs=8)
        print(f"Reward: {tree.reward} +- {tree.std_reward}, Size: {tree.get_tree_size()}, SR: {tree.success_rate}")
        print(f"-------------------")