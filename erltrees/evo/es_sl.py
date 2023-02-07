import argparse
from datetime import datetime
import json
import pdb
import signal
import sys
import time
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
# 'Essentials of Metaheuristics' (Sean Luke, 2016)

class EvolutionaryStrategySL(EvolutionaryAlgorithm):
    def __init__(self, tournament_size=0, **kwargs):
        super(EvolutionaryStrategySL, self).__init__(**kwargs)
        self.tournament_size = tournament_size
    
    def run(self, generations, initial_pop=None,
            should_plot=False, should_save_plot=False, 
            should_render=False, render_every=None):
        
        # Seeding initial population
        population = evo.fill_initial_pop(self.config, initial_pop, self.lamb, self.initial_depth,
            (0 if args["should_attenuate_alpha"] else args["alpha"]),
            jobs_to_parallelize=self.jobs_to_parallelize, should_penalize_std=self.should_penalize_std,
            should_norm_state=self.should_norm_state, episodes=100)
        population = [i.copy() for i in population]
        self.allbest = population[np.argmax([i.fitness for i in population])]

        for generation in range(generations):
            current_alpha = self.calc_alpha(population, generation, generations)
            
            # Parent selection (mu fittest individuals)
            population.sort(key=lambda x : x.fitness, reverse=True)
            parent_population = population[:self.mu]
            
            # Offspring generation
            child_population = []
            for parent in parent_population:
                for _ in range(self.lamb // self.mu):
                    child = parent.copy()
                    mutate(child, mutation=self.mutation_type)
                    child_population.append(child)
            
            # Evaluating candidate population (only children)
            candidate_population = child_population
            rl.fill_metrics(self.config, candidate_population, alpha=current_alpha,
                episodes=self.fit_episodes, should_norm_state=self.should_norm_state,
                penalize_std=self.should_penalize_std, n_jobs=self.jobs_to_parallelize)

            # Survivor selection (truncation selection)
            population = candidate_population
            if self.should_include_allbest:
                population += [self.allbest.copy()]

            if self.should_attenuate_alpha and self.allbest:
                # in this case, fitness of best individual needs to be updated
                self.allbest.fitness = rl.calc_fitness_tree(self.allbest, current_alpha, self.should_penalize_std)

            # Housekeeping
            self.save_allbest(current_alpha, population)
            self.increment_log_history(population)
            if self.verbose:
                self.print_last_metrics()

        # if self.should_prune_best_by_visits:
        #     rl.collect_and_prune_by_visits(
        #         self.allbest, threshold=5, episodes=100, 
        #         should_norm_state=self.should_norm_state)
            
        self.plot_metrics(should_plot, should_save_plot)
        # self.allbest = self.evaluate_popbests(candidate_pool_size=10, verbose=self.verbose)
        rl.fill_metrics(self.config, [self.allbest], self.alpha, 1000, 
            self.should_norm_state, self.should_penalize_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary Programming')
    parser.add_argument('-t', '--task', help="Which task to run?", required=True)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-g', '--generations', help="Number of generations", required=True, type=int)
    parser.add_argument('-o', '--output_path', help="Path to save files", required=False, default=None, type=str)
    parser.add_argument('-i', '--initial_pop', help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--mu', help="Value of mu", required=True, type=int)
    parser.add_argument('--lambda', help="Value of lambda", required=True, type=int)
    parser.add_argument('--tournament_size', help="Size of tournament", required=False, default=0, type=int)
    parser.add_argument('--mutation_type', help="Type of mutation", required=True, default="A", type=str)
    parser.add_argument('--initial_depth', help="Randomly initialize the algorithm with trees of what depth?", required=True, type=int)
    parser.add_argument('--mutation_qt', help="How many mutations to execute?", required=False, default=1, type=int)
    parser.add_argument('--alpha', help="How to penalize tree size?", required=True, type=float)
    parser.add_argument('--should_norm_state', help="Should normalize state?", required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_prune_by_visits', help='Should prune every tree by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_prune_best_by_visits', help='Should prune best tree by visits?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_recheck_popbest', help='Should recheck population best?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_include_allbest', help='Should always include all-time best individual in the population?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--recheck_popbest_episodes', help='How many episodes to run a recheck on popbest?', required=False, default=100, type=int)
    parser.add_argument('--should_attenuate_alpha', help='Should attenuate alpha?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_penalize_std', help='Should penalize standard deviation?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--episodes', help='Number of episodes to run when evaluating model', required=False, default=10, type=int)
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save_plot', help='Should save plot performance?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--jobs_to_parallelize', help='How many jobs to parallelize?', required=False, default=-1, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = configs.get_config(args["task"])
    
    TIME_START = time.time()

    command_line = str(args)
    command_line += "\n\npython -m erltrees.evo.es_sl " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    output_path = ("data/log_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt") if args['output_path'] in [None, "None", ""] else args['output_path']
    print(f"{command_line}")
    print(f"output_path: '{output_path}'")
    io.save_history_to_file(config, None, output_path, prefix=command_line)

    # Setting up initial population
    initial_pop = evo.get_initial_pop(config, 
        alpha=(0 if args["should_attenuate_alpha"] else args["alpha"]),
        popsize=args["lambda"], depth_random_indiv=args["initial_depth"],
        should_penalize_std=args["should_penalize_std"], should_norm_state=args["should_norm_state"], 
        episodes=100, initial_pop=args["initial_pop"], n_jobs=args["jobs_to_parallelize"])    
    
    rl.fill_metrics(config, initial_pop, alpha=(0 if args["should_attenuate_alpha"] else args["alpha"]), 
        episodes=100, should_norm_state=args["should_norm_state"],
        penalize_std=args["should_penalize_std"], n_jobs=args["jobs_to_parallelize"])

    # Running simulations
    sim_history = []
    es = None

    def handler(sig, frame):
        print(f"Saving and exiting... Output path: {output_path}")
        sim_history.append((es.allbest, es.allbest.reward, es.allbest.get_tree_size(), None))
        io.save_history_to_file(config, sim_history, output_path, prefix=command_line + "\n(Interrupted)\n\n")
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)

    for _ in range(args['simulations']):
        # Executing EA
        es = EvolutionaryStrategySL(tournament_size=args["tournament_size"], 
            config=config,
            mu=args["mu"], lamb=args["lambda"], alpha=args["alpha"], 
            initial_depth=args["initial_depth"], fit_episodes=args["episodes"],
            mutation_type=args["mutation_type"], should_norm_state=args["should_norm_state"],
            should_penalize_std=args["should_penalize_std"], 
            should_recheck_popbest=args["should_recheck_popbest"],
            should_include_allbest=args["should_include_allbest"],
            recheck_popbest_episodes=args["recheck_popbest_episodes"],
            should_prune_by_visits=args["should_prune_by_visits"],
            should_attenuate_alpha=args["should_attenuate_alpha"],
            should_prune_best_by_visits=args["should_prune_best_by_visits"],
            jobs_to_parallelize=args["jobs_to_parallelize"], 
            verbose=args["verbose"])
        es.run(args['generations'], initial_pop, args['should_plot'], args['should_save_plot'])

        # Logging results
        sim_history.append((es.allbest, es.allbest.reward, es.allbest.get_tree_size(), None))

        # Printing results
        print(f"Simulations run until now: {len(sim_history)} / {args['simulations']}")
        print(sim_history)
        print(f"output_path: '{output_path}'")

        TIME_END = time.time()
        io.save_history_to_file(config, sim_history, output_path, 
            elapsed_time=TIME_END-TIME_START, prefix=command_line)
        es.save_history_as_csv()