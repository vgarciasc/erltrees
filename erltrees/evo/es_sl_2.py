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
# 'Essentials of Metaheuristics' (Sean Luke, 2016)

class EvolutionaryStrategySL2(EvolutionaryAlgorithm):
    def __init__(self, tournament_size=0, **kwargs):
        super(EvolutionaryStrategySL2, self).__init__(**kwargs)
        self.tournament_size = tournament_size
    
    def run(self, generations, initial_pop=None,
            should_plot=False, should_save_plot=False, 
            should_render=False, render_every=None):
        
        # Seeding initial population
        initial_pop = self.get_initial_pop(self.lamb, self.initial_depth, episodes=100, filename=initial_pop)
        population = [i.copy() for i in initial_pop]
        best = population[np.argmax([i.fitness for i in population])]
        self.allbest = best
        evaluations_to_success = 0

        # Main loop
        for generation in range(generations):
            current_alpha = self.calc_alpha(population, generation, generations)

            # Parent selection
            population.sort(key=lambda x : x.fitness, reverse=True)
            parent_population = population[:self.mu]

            # Offspring generation
            child_population = []
            for parent in parent_population:
                for _ in range(self.lamb // self.mu):
                    child = parent.copy()
                    mutate(child, mutation=self.mutation_type)
                    child_population.append(child)

            rl.fill_metrics(self.config, child_population, alpha=current_alpha, 
                episodes=self.fit_episodes, should_norm_state=self.should_norm_state, 
                penalize_std=self.should_penalize_std, n_jobs=self.jobs_to_parallelize)
            
            # Survivor selection
            population = child_population

            fitnesses = [i.fitness for i in population]
            individual_max_fitness = population[np.argmax(fitnesses)]

            if self.should_attenuate_alpha:
                # in this case, fitness of best individual needs to be updated
                best.fitness = rl.calc_fitness(best.reward, best.std_reward, best.get_tree_size(), current_alpha, self.should_penalize_std)
            
            if individual_max_fitness.fitness > best.fitness:
                rl.fill_metrics(config, [individual_max_fitness], 
                    current_alpha, episodes=100, 
                    should_norm_state=self.should_norm_state, 
                    penalize_std=self.should_penalize_std, 
                    n_jobs=self.jobs_to_parallelize)
                print(f"Individual max fitness (rechecked): (reward: {'{:.3f}'.format(individual_max_fitness.reward)}, fitness: {'{:.3f}'.format(individual_max_fitness.fitness)})")
                if individual_max_fitness.fitness > best.fitness:
                    best = individual_max_fitness.copy()
                    self.allbest = best
                    if str(best) != str(individual_max_fitness):
                        evaluations_to_success = generation

            self.increment_log_history(population)
            self.print_last_metrics()
        
        io.printv(f"[yellow]Best individual w/ reward {best.reward}:", self.verbose)
        io.printv(best, self.verbose)

        if should_save_plot:
            figure_file = "data/plots/" + config['name'] + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".png"
            print(f"Saving figure to '{figure_file}'.")
            plt.savefig(figure_file)

        if self.should_prune_best_by_visits:
            rl.collect_and_prune_by_visits(
                best, threshold=5, episodes=100, 
                should_norm_state=self.should_norm_state)

        self.plot_metrics(should_plot, should_save_plot)

        return best, best.reward, best.get_tree_size(), evaluations_to_success

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
    
    command_line = str(args)
    command_line += "\n\npython -m erltrees.evo.es_sl_2 " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    output_path = ("data/log_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S") + ".txt") if args['output_path'] in [None, ""] else args['output_path']
    print(f"{command_line}")
    print(f"output_path: '{output_path}'")
    io.save_history_to_file(config, None, output_path, prefix=command_line)

    history = []
    for _ in range(args['simulations']):
        es = EvolutionaryStrategySL2(tournament_size=args["tournament_size"], 
            config=config,
            mu=args["mu"], lamb=args["lambda"], alpha=args["alpha"], 
            initial_depth=args["initial_depth"], fit_episodes=args["episodes"],
            mutation_type=args["mutation_type"], should_norm_state=args["should_norm_state"],
            should_penalize_std=args["should_penalize_std"], 
            should_recheck_popbest=args["should_recheck_popbest"],
            recheck_popbest_episodes=args["recheck_popbest_episodes"],
            should_prune_by_visits=args["should_prune_by_visits"],
            should_attenuate_alpha=args["should_attenuate_alpha"],
            should_prune_best_by_visits=args["should_prune_best_by_visits"],
            jobs_to_parallelize=args["jobs_to_parallelize"], 
            verbose=args["verbose"])

        initial_pop = []
        if args['initial_pop'] != '':
            with open(args['initial_pop']) as f:
                json_obj = json.load(f)
            initial_pop = [evo_tree.Individual.read_from_string(config, json_str) 
                for json_str in json_obj]

        tree, reward, size, evals2suc = es.run(args['generations'], initial_pop, args['should_plot'], args['should_save_plot'])

        reward, _, _ = rl.collect_metrics(
            tree.config, [tree],
            episodes=100,
            should_norm_state=args['should_norm_state'])[0]
        history.append((tree, reward, size, evals2suc))
        print(f"Simulations run until now: {len(history)} / {args['simulations']}")
        print(history)
        print(f"output_path: '{output_path}'")
        io.save_history_to_file(config, history, output_path, prefix=command_line)

    trees, rewards, sizes, evals2suc = zip(*history)
    trees = np.array(trees)
    
    rl.fill_metrics(config, trees, args['alpha'], episodes=1000, 
        should_norm_state=args['should_norm_state'], penalize_std=False)
    
    successes = [1 if e > 0 else 0 for e in evals2suc]
    evals2suc = [e for e in evals2suc if e > 0]
    
    if args["verbose"]:
        io.console.rule(f"[bold red]Hall of Fame")
        print(f"[green][bold]5 best trees:[/bold][/green]")
        sorted(trees, key=lambda x: x.fitness, reverse=True)
        for i, tree in enumerate(trees[:5]):
            print(f"#{i}: (reward {tree.reward}, size {tree.get_tree_size()})")
            print(tree)

        io.console.rule(f"[bold red]RESULTS")
    print(f"[green][bold]Mean Best Reward[/bold][/green]: {np.mean(rewards)}")
    print(f"[green][bold]Mean Best Size[/bold][/green]: {np.mean(sizes)}")
    print(f"[green][bold]Average Evaluations to Success[/bold][/green]: {np.mean(evals2suc)}")
    print(f"[green][bold]Success Rate[/bold][/green]: {np.mean(successes)}")

    io.save_history_to_file(config, history, output_path, prefix=command_line)