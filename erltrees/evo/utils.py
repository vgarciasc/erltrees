import json
import pdb
import numpy as np
import erltrees.evo.evo_tree as evo_tree
import erltrees.rl.utils as rl
from copy import deepcopy

def initialize_population(config, initial_depth, popsize, initial_pop, alpha, 
    episodes, norm_state, should_penalize_std, jobs_to_parallelize):
    
    population = []
    if initial_pop != []:
        for tree in initial_pop: #assuming initial pop of EvoTreeNodes
            tree = tree.copy()
            if norm_state:
                tree.normalize_thresholds()
            population.append(tree)

    for _ in range(len(population), popsize):
        population.append(evo_tree.Individual.generate_random_tree(
            config, depth=initial_depth))
    
    rl.fill_metrics(config, population, alpha=alpha, 
        episodes=episodes, should_norm_state=norm_state,
        task_solution_threshold=config["task_solution_threshold"],
        penalize_std=should_penalize_std, 
        n_jobs=jobs_to_parallelize)
    
    return population

def get_initial_pop(config, alpha, popsize, depth_random_indiv, should_penalize_std=False, 
    should_norm_state=True, episodes=10, initial_pop=None, n_jobs=-1):
    
    # Initialize from file
    population = []

    if type(initial_pop) == str:
        with open(initial_pop) as f:
            json_obj = json.load(f)
        
        population = [evo_tree.Individual.read_from_string(config, json_str) for json_str in json_obj]
    elif initial_pop is not None:
        population = [deepcopy(x) for x in initial_pop]
    
    if should_norm_state:
        for p in population:
            p.normalize_thresholds()
        
    # for _ in range(len(population), popsize):
    #     population.append(evo_tree.Individual.generate_random_tree(config, depth_random_indiv))

    return population

def fill_initial_pop(config, population, popsize, initial_depth, alpha, jobs_to_parallelize,
        should_penalize_std=False, should_norm_state=True, episodes=10):

    new_population = []
    # Fill with the rest
    for _ in range(len(population), popsize):
        tree = evo_tree.Individual.generate_random_tree(
            config, depth=initial_depth)
        new_population.append(tree)
    
    rl.fill_metrics(config, new_population, alpha=alpha, 
        episodes=episodes, should_norm_state=should_norm_state,
        penalize_std=should_penalize_std,
        task_solution_threshold=config["task_solution_threshold"],
        n_jobs=jobs_to_parallelize)
    
    return population + new_population

def tournament_selection(population, q):
    candidates = np.random.choice(population, size=q, replace=False)
    return max(candidates, key=lambda x : x.fitness)

def print_population_ids(population, selected_population):
    print([population.index(i) for i in selected_population])

def print_population_rewards(population):
    print([i.reward for i in population])