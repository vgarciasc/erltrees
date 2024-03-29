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

# Implementation of (λ + μ) Evolutionary Strategy as defined in 
# Eiben and Smith

class EvolutionaryStrategy(EvolutionaryAlgorithm):
    def __init__(self, tournament_size=0, **kwargs):
        super(EvolutionaryStrategy, self).__init__(**kwargs)
        self.tournament_size = tournament_size
    
    def run(self, generations, initial_pop_filename=None,
            should_plot=False, should_save_plot=False, 
            should_render=False, render_every=None):
        
        # Seeding initial population
        initial_pop = self.get_initial_pop(self.lamb, self.initial_depth, initial_pop_filename)
        population = [i.copy() for i in initial_pop]

        for generation in range(generations):
            current_alpha = self.calc_alpha(generation, generations)

            # Parent selection (mu fittest individuals)
            parent_population = []
            for _ in range(self.mu):
                parent = evo.tournament_selection(population, self.tournament_size)
                parent_population.append(parent)
            
            # Offspring generation
            child_population = []
            for parent in parent_population:
                for _ in range(self.lamb // self.mu):
                    child = parent.copy()
                    mutate(child, mutation=self.mutation)
                    child_population.append(child)
            
            # Evaluating candidate population (only children)
            candidate_population = parent_population + child_population
            rl.fill_metrics(self.config, candidate_population, alpha=current_alpha,
                episodes=self.fit_episodes, should_norm_state=self.should_norm_state,
                penalize_std=self.should_penalize_std, n_jobs=self.jobs_to_parallelize)

            # Survivor selection (truncation selection)
            candidate_population.sort(key=lambda x : x.fitness, reverse=True)
            population = population[:self.lamb]

            # Housekeeping
            self.save_allbest(population)
            self.increment_log_history(population)
            if self.verbose:
                self.print_last_metrics()
            
        self.plot_metrics(should_plot, should_save_plot)
        self.allbest = self.evaluate_popbests(candidate_pool_size=10, verbose=self.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary Programming')
    parser.add_argument('-t', '--task', help="Which task to run?", required=True)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-g', '--generations', help="Number of generations", required=True, type=int)
    parser.add_argument('-o', '--output_path', help="Path to save files", required=False, default=None, type=str)
    parser.add_argument('-i', '--initial_pop', help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--mu', help="Value of mu", required=True, type=int)
    parser.add_argument('--lambda', help="Value of lambda", required=True, type=int)
    parser.add_argument('--tournament_size', help="Size of tournament", required=True, type=int)
    parser.add_argument('--mutation_type', help="Type of mutation", required=True, default="A", type=str)
    parser.add_argument('--initial_depth', help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--mutation_qt', help="How many mutations to execute?", required=False, default=1, type=int)
    parser.add_argument('--alpha', help="How to penalize tree size?", required=True, type=float)
    parser.add_argument('--should_norm_state', help="Should normalize state?", required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_attenuate_alpha', help='Should attenuate alpha?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_recheck_popbest', help='Should recheck population best?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_prune_by_visits', help='Should prune every tree by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
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
    command_line += "\n\npython -m erltrees.evo.es " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
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

        es = EvolutionaryStrategy(tournament_size=args["tournament_size"], 
            config=config,
            mu=args["mu"], lamb=args["lambda"], alpha=args["alpha"], 
            initial_depth=args["initial_depth"], fit_episodes=args["episodes"],
            mutation_type=args["mutation_type"], should_norm_state=args["should_norm_state"],
            should_penalize_std=args["should_penalize_std"], 
            should_recheck_popbest=args["should_recheck_popbest"],
            should_attenuate_alpha=args["should_attenuate_alpha"],
            should_prune_by_visits=args["should_prune_by_visits"],
            should_prune_best_by_visits=args["should_prune_best_by_visits"],
            jobs_to_parallelize=args["jobs_to_parallelize"], 
            verbose=args["verbose"])
        
        es.run(args['generations'], initial_pop, args['should_plot'], args['should_save_plot'])

        sim_history.append((es.allbest, es.allbest.reward, es.allbest.get_tree_size()))

        print(f"Simulations run until now: {len(sim_history)} / {args['simulations']}")
        print(sim_history)
        print(f"output_path: '{output_path}'")
        io.save_history_to_file(config, sim_history, output_path, prefix=command_line)