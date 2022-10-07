import numpy as np
import erltrees.evo.evo_tree as evo_tree
import erltrees.rl.utils as rl

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
        penalize_std=should_penalize_std, 
        n_jobs=jobs_to_parallelize)
    
    return population

def tournament_selection(population, q):
    candidates = np.random.choice(population, size=q, replace=False)
    return max(candidates, key=lambda x : x.fitness)

def print_population_ids(population, selected_population):
    print([population.index(i) for i in selected_population])

def print_population_rewards(population):
    print([i.reward for i in population])