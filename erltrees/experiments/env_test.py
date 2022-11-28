import argparse 
import pdb
import gym

from erltrees.evo.evo_tree import Individual
import erltrees.rl.utils as rl
import erltrees.rl.configs as configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Pruning')
    parser.add_argument('--task', help="Which task to run?", type=str, required=True)
    args = vars(parser.parse_args())

    config = configs.get_config(args["task"])
    tree = Individual.generate_random_tree(config, depth=2)

    rl.collect_metrics(config, [tree], alpha=0.0, episodes=25, render=True)