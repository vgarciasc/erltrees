import argparse
from datetime import datetime
import json
import pdb
from rich import print
import numpy as np
import matplotlib.pyplot as plt
from erltrees.evo.evo_algorithm import EvolutionaryAlgorithm

import erltrees.evo.evo_tree as evo_tree
from erltrees.evo.mutations import mutate
import erltrees.evo.utils as evo
import erltrees.io as io
import erltrees.rl.utils as rl
import erltrees.rl.configs as configs

# Implementation of (λ, μ) Evolutionary Strategy as defined in 
# Eiben and Smith

class ReducedVNS(EvolutionaryAlgorithm):
    def __init__(self, perturbation_patience, **kwargs):
        super(ReducedVNS, self).__init__(**kwargs)
        self.perturbation_patience = perturbation_patience
    
    def run(self, generations, initial_pop_filename=None,
            should_plot=False, should_save_plot=False, 
            should_render=False, render_every=None):
        
        initial_pop = self.get_initial_pop(self.lamb, self.initial_depth, initial_pop_filename)
        initial_pop.sort(key=lambda x : x.fitness, reverse=True)
        solution = initial_pop[0].copy()

        history = []
        best_solution = None

        k = 1
        for generation in range(generations):
            current_alpha = self.calc_alpha(generation, generations)

            candidate_solution = solution.copy()
            for _ in range(1 + k // self.perturbation_patience):
                mutate(candidate_solution, mutation=self.mutation)
        
            # mutate(candidate_solution, mutation=self.mutation)

            rl.fill_metrics(self.config, [candidate_solution], alpha=current_alpha,
                episodes=self.fit_episodes, should_norm_state=self.should_norm_state,
                penalize_std=self.should_penalize_std, n_jobs=self.jobs_to_parallelize)
            
            history.append((candidate_solution.reward, candidate_solution.std_reward, solution.reward, solution.std_reward))

            if candidate_solution.fitness > solution.fitness:
                solution = candidate_solution
                k = 1
            else:
                k += 1
                if np.random.uniform(0, 1) < np.exp(- (solution.fitness - candidate_solution.fitness)):
                    solution = candidate_solution
                    k = 1
            
            if best_solution is None or candidate_solution.fitness > best_solution.fitness:
                best_solution = candidate_solution.copy()

            io.printv(f"[white]Iteration #{generation}[/white]", self.verbose)
            io.printv(f"[green]BEST: Fitness: {'{:.3f}'.format(best_solution.fitness)}, Reward: {'{:.3f}'.format(best_solution.reward)} ± {'{:.3f}'.format(best_solution.std_reward)}, Size: {solution.get_tree_size()}[/green]", self.verbose)
            io.printv(f"[green]CURR: Fitness: {'{:.3f}'.format(solution.fitness)}, Reward: {'{:.3f}'.format(solution.reward)} ± {'{:.3f}'.format(solution.std_reward)}, Size: {solution.get_tree_size()}[/green]", self.verbose)
            io.printv(f"[yellow]ITER: Fitness: {'{:.3f}'.format(candidate_solution.fitness)}, Reward: {'{:.3f}'.format(candidate_solution.reward)} ± {'{:.3f}'.format(candidate_solution.std_reward)}, Size: {solution.get_tree_size()}[/yellow]", self.verbose)
            io.printv("[white]---------------[/white]", self.verbose)

        curr_rewards, curr_std_rewards, rewards, std_rewards = zip(*history)
        curr_rewards = np.array(curr_rewards)
        curr_std_rewards = np.array(curr_std_rewards)
        rewards = np.array(rewards)
        std_rewards = np.array(std_rewards)
        plt.figure(figsize=(18,12))
        plt.plot(range(generations), curr_rewards, color='blue', linewidth=1, label="Iteration reward")
        plt.plot(range(generations), rewards, color='green', linewidth=2, label="Best reward")
        plt.fill_between(range(generations), curr_rewards - curr_std_rewards, curr_rewards + curr_std_rewards, color="blue", alpha=0.2)
        plt.fill_between(range(generations), rewards - std_rewards, rewards + std_rewards, color="green", alpha=0.2)
        filename = "data/plots/RVNS_" + self.config['name'] + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".png"
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.savefig(filename)

        return solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary Programming')
    parser.add_argument('-t', '--task', help="Which task to run?", required=True)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-g', '--generations', help="Number of generations", required=True, type=int)
    parser.add_argument('-o', '--output_path', help="Path to save files", required=False, default=None, type=str)
    parser.add_argument('-i', '--initial_pop', help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--mu', help="Value of mu", required=True, type=int)
    parser.add_argument('--lambda', help="Value of lambda", required=True, type=int)
    parser.add_argument('--perturbation_patience', help="Number of mutations is incremented every 'perturbation_patience' iterations", required=False, default=10, type=int)
    parser.add_argument('--mutation_type', help="Type of mutation", required=True, default="A", type=str)
    parser.add_argument('--initial_depth', help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--mutation_qt', help="How many mutations to execute?", required=False, default=1, type=int)
    parser.add_argument('--alpha', help="How to penalize tree size?", required=True, type=float)
    parser.add_argument('--should_norm_state', help="Should normalize state?", required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_prune_by_visits', help='Should prune every tree by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_attenuate_alpha', help='Should attenuate alpha?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_prune_best_by_visits', help='Should prune best tree by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_penalize_std', help='Should penalize standard deviation?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save_plot', help='Should save plot performance?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--jobs_to_parallelize', help='How many jobs to parallelize?', required=False, default=-1, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = configs.get_config(args["task"])
    
    command_line = str(args)
    command_line += "\n\npython -m erltrees.evo.rvns " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    output_path = ("data/log_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt") if args['output_path'] in [None, ""] else args['output_path']
    print(f"{command_line}")
    print(f"output_path: '{output_path}'")
    io.save_history_to_file(config, None, output_path, prefix=command_line)

    sim_history = []
    for _ in range(args['simulations']):
        initial_pop = []
        if args['initial_pop'] != '':
            with open(args['initial_pop']) as f:
                json_obj = json.load(f)
            initial_pop = [evo_tree.Individual.read_from_string(config, json_str) for json_str in json_obj]

        es = ReducedVNS(perturbation_patience=args["perturbation_patience"],
            config=config, mu=args["mu"], lamb=args["lambda"], alpha=args["alpha"], 
            initial_depth=args["initial_depth"], fit_episodes=args["episodes"],
            mutation_type=args["mutation_type"], should_norm_state=args["should_norm_state"],
            should_penalize_std=args["should_penalize_std"], 
            should_prune_by_visits=args["should_prune_by_visits"],
            should_attenuate_alpha=args["should_attenuate_alpha"],
            should_prune_best_by_visits=args["should_prune_best_by_visits"],
            jobs_to_parallelize=args["jobs_to_parallelize"], 
            verbose=args["verbose"])
        
        allbest = es.run(args['generations'], initial_pop, args['should_plot'], args['should_save_plot'])

        print(allbest)