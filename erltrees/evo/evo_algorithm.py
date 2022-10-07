from datetime import datetime
import json
import pdb
from matplotlib import pyplot as plt
import numpy as np

import erltrees.io as io
import erltrees.evo.evo_tree as evo_tree
import erltrees.rl.utils as rl

class EvolutionaryAlgorithm():
    def __init__(self, config, mu, lamb, alpha, initial_depth, fit_episodes=10, 
                 mutation_type="A", tournament_size=0,
                 should_norm_state=True,
                 should_penalize_std=True,
                 should_attenuate_alpha=False,
                 should_prune_by_visits=False,
                 recheck_popbest_episodes=False,
                 should_recheck_popbest=False,
                 should_prune_best_by_visits=False,
                 jobs_to_parallelize=-1,
                 verbose=False, **kwargs):

        self.config = config
        self.mu = mu
        self.lamb = lamb
        self.alpha = alpha
        self.initial_depth = initial_depth
        self.fit_episodes = fit_episodes
        self.mutation_type = mutation_type
        self.tournament_size = tournament_size
        self.should_attenuate_alpha = should_attenuate_alpha
        self.should_recheck_popbest = should_recheck_popbest
        self.recheck_popbest_episodes = recheck_popbest_episodes
        self.should_norm_state = should_norm_state
        self.should_penalize_std = should_penalize_std
        self.should_prune_by_visits = should_prune_by_visits
        self.should_prune_best_by_visits = should_prune_best_by_visits
        self.jobs_to_parallelize = jobs_to_parallelize
        self.verbose = verbose

        self.history = []
        self.popbests = []
        self.allbest = None
    
    def calc_alpha(self, population, curr_gen, total_gens):
        avg_reward = np.mean([i.reward for i in population])
        curr_alpha = self.alpha * (curr_gen / total_gens) if self.should_attenuate_alpha else self.alpha
        curr_alpha = 0 if avg_reward == self.config["min_score"] else curr_alpha
        return curr_alpha

    def save_allbest(self, alpha, population):
        fitnesses = [i.fitness for i in population]
        popbest = population[np.argmax(fitnesses)]

        if self.should_attenuate_alpha:
            # in this case, fitness of best individual needs to be updated
            self.allbest.fitness = rl.calc_fitness(
                self.allbest.reward, self.allbest.std_reward,
                self.allbest.get_tree_size(), alpha,
                self.should_penalize_std)
        
        if popbest.fitness > self.allbest.fitness:
            if self.should_recheck_popbest:
                rl.fill_metrics(self.config, [popbest], alpha, 
                    episodes=self.recheck_popbest_episodes,
                    should_norm_state=self.should_norm_state, 
                    penalize_std=self.should_penalize_std, 
                    n_jobs=self.jobs_to_parallelize)
                
                print(f"Individual max fitness (rechecked): (reward: {'{:.3f}'.format(popbest.reward)}, " + \
                    f"fitness: {'{:.3f}'.format(popbest.fitness)})")

                if popbest.fitness > self.allbest.fitness:
                    self.allbest = popbest.copy()
            else:
                self.allbest = popbest.copy()
    
    def evaluate_popbests(self, candidate_pool_size=10, verbose=False):
        if len(self.popbests) == 0:
            raise "Array of best individuals from each generation is empty."

        candidates = [(i, popbest) for (i, popbest) in enumerate(self.popbests)]
        
        if self.should_prune_best_by_visits:
            for i, candidate in candidates:
                rl.collect_and_prune_by_visits(candidate, 
                    should_norm_state=self.should_norm_state)

        candidates.sort(key=lambda x : x[1].fitness, reverse=True)
        candidates = candidates[:candidate_pool_size]

        ids, trees = zip(*candidates)

        rl.fill_metrics(self.config, trees, self.alpha, 100, 
            self.should_norm_state, self.should_penalize_std, 
            n_jobs=self.jobs_to_parallelize)
        
        candidates = [(ids[i], tree) for (i, tree) in enumerate(trees)]
        candidates.sort(key=lambda x : x[1].fitness, reverse=True)

        for rank, (id, candidate) in enumerate(candidates):
            io.printv(f"#{rank} (gen {id}): Fitness ({'{:.3f}'.format(candidate.fitness)}), Reward: ({'{:.3f}'.format(candidate.reward)} +- {'{:.3f}'.format(candidate.std_reward)}), Size: {candidate.get_tree_size()}", verbose)
        
        _, allbest = candidates[0]
        return allbest

    def get_population_metrics(self, population):
        rewards = [i.reward for i in population]
        fitnesses = [i.fitness for i in population]
        tree_sizes = [i.get_tree_size() for i in population]
        popbest = population[np.argmax(fitnesses)]

        popbest_fitness = popbest.fitness
        popbest_avg_reward = popbest.reward
        popbest_std_reward = popbest.std_reward
        popbest_size = popbest.get_tree_size()
        allbest_fitness = self.allbest.fitness
        allbest_reward = self.allbest.reward
        allbest_std_reward = self.allbest.std_reward
        allbest_size = self.allbest.get_tree_size()
        pop_avg_fitness = np.mean(fitnesses)
        pop_std_fitness = np.mean(fitnesses)
        pop_avg_reward = np.mean(rewards)
        pop_std_reward = np.std(rewards)
        pop_avg_size = np.mean(tree_sizes)
        pop_std_size = np.std(tree_sizes)

        fitness_metrics = (popbest_fitness, pop_avg_fitness, pop_std_fitness, allbest_fitness)
        reward_metrics = (popbest_avg_reward, popbest_std_reward, pop_avg_reward, 
            pop_std_reward, allbest_reward, allbest_std_reward)
        size_metrics = (popbest_size, pop_avg_size, pop_std_size, allbest_size)

        return fitness_metrics, reward_metrics, size_metrics

    def increment_log_history(self, population):
        fitness_metrics, reward_metrics, size_metrics = self.get_population_metrics(population)
        self.history.append((fitness_metrics, reward_metrics, size_metrics))
    
    def print_last_metrics(self):
        if not self.verbose:
            return

        generation = len(self.history)        
        io.console.rule(f"[bold red]Generation #{generation}")

        fitness_metrics, reward_metrics, size_metrics = self.history[-1]
        (popbest_fitness, pop_avg_fitness, pop_std_fitness, allbest_fitness) = fitness_metrics
        (popbest_avg_reward, popbest_std_reward, pop_avg_reward, 
            pop_std_reward, allbest_reward, allbest_std_reward) = reward_metrics
        (popbest_size, pop_avg_size, pop_std_size, allbest_size) = size_metrics

        io.printv(f"[underline]Fitness[/underline]: {{[green]All best: {'{:.3f}'.format(allbest_fitness)}[/green], " + \
                f"[yellow]Gen best: {'{:.3f}'.format(popbest_fitness)}[/yellow], " + \
                f"[grey]Avg: {'{:.3f}'.format(pop_avg_fitness)} ± {'{:.3f}'.format(pop_std_fitness)}[/grey]}}", self.verbose)
        io.printv(f"[underline]Reward [/underline]: {{[green]All best: {'{:.3f}'.format(allbest_reward)} ± {'{:.3f}'.format(allbest_std_reward)}[/green], " + \
                f"[yellow]Gen best: {'{:.3f}'.format(popbest_avg_reward)} ± {'{:.3f}'.format(popbest_std_reward)}[/yellow], " + \
                f"[grey]Avg: {'{:.3f}'.format(pop_avg_reward)} ± {'{:.3f}'.format(pop_std_reward)}[/grey]", self.verbose)
        io.printv(f"[underline]Size   [/underline]: {{[green]All best: {allbest_size}[/green], " + \
                f"[yellow]Gen best: {popbest_size}[/yellow], " + \
                f"[grey]Avg: {'{:.3f}'.format(pop_avg_size)} ± {'{:.3f}'.format(pop_std_size)}[/grey]}}", self.verbose)

    def plot_metrics(self, should_plot=False, should_save_plot=True):
        x = range(len(self.history))

        fitnesses, rewards, sizes = zip(*self.history)
        popbest_fitnesses, pop_avg_fitnesses, pop_std_fitnesses, allbest_fitnesses = zip(*fitnesses)
        popbest_avg_rewards, popbest_std_rewards, pop_avg_rewards, pop_std_rewards, allbest_rewards, allbest_std_rewards = zip(*rewards)
        popbest_sizes, pop_avg_sizes, pop_std_sizes, allbest_sizes = zip(*sizes)

        popbest_fitnesses = np.array(popbest_fitnesses)
        pop_avg_fitnesses = np.array(pop_avg_fitnesses)
        pop_std_fitnesses = np.array(pop_std_fitnesses)
        allbest_fitnesses = np.array(allbest_fitnesses)
        popbest_avg_rewards = np.array(popbest_avg_rewards)
        popbest_std_rewards = np.array(popbest_std_rewards)
        pop_avg_rewards = np.array(pop_avg_rewards)
        pop_std_rewards = np.array(pop_std_rewards)
        allbest_rewards = np.array(allbest_rewards)
        allbest_std_rewards = np.array(allbest_std_rewards)
        popbest_sizes = np.array(popbest_sizes)
        pop_avg_sizes = np.array(pop_avg_sizes)
        pop_std_sizes = np.array(pop_std_sizes)
        allbest_sizes = np.array(allbest_sizes)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))
        ax1.clear()
        ax1.plot(x, popbest_avg_rewards, color="green", label="Generation best reward")
        ax1.fill_between(x, popbest_avg_rewards - popbest_std_rewards, popbest_avg_rewards + popbest_std_rewards, color="green", alpha=0.2)
        ax1.plot(x, pop_avg_rewards, color="blue", label="Generation average reward")
        ax1.fill_between(x, pop_avg_rewards - pop_std_rewards, pop_avg_rewards + pop_std_rewards, color="blue", alpha=0.2)
        ax1.plot(x, allbest_rewards, color="black", linestyle="dashed", label="Overall best tree's reward")
        ax1.fill_between(x, allbest_rewards - allbest_std_rewards, allbest_rewards + allbest_std_rewards, color="black", alpha=0.2)
        ax2.clear()
        ax2.plot(x, popbest_sizes, color="red", label="Generation best size")
        ax2.plot(x, pop_avg_sizes, color="orange", label="Generation average size")
        ax2.plot(x, allbest_sizes, color="black", linestyle="dashed", label="Overall best tree's size")
        ax2.fill_between(x, pop_avg_sizes - pop_std_sizes, pop_avg_sizes + pop_std_sizes, color="orange", alpha=0.2)
        
        ax2.set_xlabel("Generations")
        ax1.set_ylabel("Fitness")
        ax2.set_ylabel("Tree size")
        ax1.legend()
        ax2.legend()

        if should_plot:
            plt.show()
        
        if should_save_plot:
            figure_file = "data/plots/" + self.config['name'] + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".png"
            print(f"Saving figure to '{figure_file}'.")
            plt.savefig(figure_file)
    
    def get_initial_pop(self, popsize, initial_depth, filename=None):
        # Initialize from file
        file_pop = []
        if filename:
            with open(filename) as f:
                json_obj = json.load(f)
            
            file_pop = [evo_tree.Individual.read_from_string(self.config, json_str) for json_str in json_obj]
        
            for tree in file_pop:
                if self.should_norm_state:
                    tree = tree.copy()
                    tree.normalize_thresholds()

        population = [] + file_pop

        # Fill with the rest
        for _ in range(len(population), popsize):
            tree = evo_tree.Individual.generate_random_tree(
                self.config, depth=initial_depth)
            population.append(tree)
        
        rl.fill_metrics(self.config, population, alpha=self.alpha, 
            episodes=self.fit_episodes, should_norm_state=self.should_norm_state,
            penalize_std=self.should_penalize_std)
        
        return population