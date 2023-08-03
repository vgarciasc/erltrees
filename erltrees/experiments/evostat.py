import json
import pdb
import time
from datetime import datetime

import numpy as np
import argparse
from joblib.parallel import Parallel, delayed
from rich.progress import Progress
from rich import print
from erltrees.evo.evo_tree import Individual
from erltrees.rl.utils import fill_metrics, collect_metrics
from erltrees.rl.configs import get_config
from erltrees.io import save_history_to_file, save_full_history_to_file
from erltrees.plots.utils import plot_full_log
import erltrees.experiments.evostat_mutations as mut


def load_initial_pop(filename, simulation, should_normalize=False):
    if filename:
        with open(filename) as f:
            json_obj = json.load(f)
        pop = [Individual.read_from_string(config, json_str) for json_str in json_obj]

        if should_normalize:
            for tree in pop:
                tree.normalize_thresholds()

        return [pop[simulation % len(pop)]]
    else:
        return []


def mutate(tree):
    operators = [mut.expand_leaf, mut.add_inner_node, mut.truncate,
                 mut.replace_child, mut.modify_leaf, mut.modify_split,
                 mut.reset_split]
    operator = np.random.choice(operators)
    operator(tree)


def evolution_strategy_step(config, alpha, episodes, lamb, mu, population, best_fitness_all, penalize_std, n_jobs=1):
    # Selecting μ parents
    population.sort(key=lambda x: x.fitness, reverse=True)
    parents = population[:mu]

    # Creating λ children
    children = []
    for parent in parents:
        for _ in range(lamb // mu):
            child = parent.copy()
            mutate(child)
            children.append(child)

    # Creating new population with μ parents and λ children
    population = parents + children

    # Evaluating new population
    fill_metrics(config, population, alpha, episodes, should_norm_state=True, penalize_std=penalize_std,
                 task_solution_threshold=config["task_solution_threshold"], n_jobs=n_jobs)

    # Saving best individual
    best_fitness, best_tree = -np.inf, None
    for individual in population:
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_tree = individual

    has_improved = False
    if best_fitness > best_fitness_all:
        has_improved = True
        best_fitness_all = best_fitness

    return population, best_fitness_all, best_tree, has_improved


def evolution_strategy(config, alpha, episodes, lamb, mu, n_gens, depth,
                       initial_pop=[], simulation_id=0, plateau_limit=None, n_jobs=1,
                       penalize_std=False, verbose=False):
    tik = time.perf_counter()

    # Setup
    last_improvement_gen_id = 0
    best_fitness = -np.inf
    best_tree = None
    full_log = []

    # Initializing population with random models
    population = initial_pop
    population += [Individual.generate_random_tree(config, depth) for _ in range(len(population), lamb)]
    fill_metrics(config, population, alpha, episodes, should_norm_state=True, penalize_std=penalize_std,
                 task_solution_threshold=config["task_solution_threshold"], n_jobs=n_jobs)

    with Progress() as progress:
        task = progress.add_task("[bold green]Running ES...", total=n_gens)

        for curr_gen in range(n_gens):
            # Evolution strategy step
            info = evolution_strategy_step(config, alpha, episodes, lamb, mu, population,
                                           best_fitness, penalize_std, n_jobs=n_jobs)
            (population, best_fitness, best_tree, has_improved) = info

            if has_improved:
                last_improvement_gen_id = curr_gen

            if ((plateau_limit is not None) and (curr_gen - last_improvement_gen_id) > plateau_limit) or (best_tree.fitness >= 495):
                break

            # Logging
            fitnesses = [ind.fitness for ind in population]
            avg_fitness = np.mean(fitnesses)
            full_log.append((fitnesses, best_tree))

            progress.update(task, advance=1, description=f"[red]Running ES...[/red] "
                                                         f"[yellow]S [gold1]{simulation_id}[/gold1] G [gold1]{curr_gen}[/gold1][/yellow] "
                                                         f"[cyan](last improv. {last_improvement_gen_id})[/cyan]"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[steel_blue]Avg. F:[/steel_blue] {'{:.2f}'.format(avg_fitness)}"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[gold1]Best F[/gold1] [grey39]([/grey39][yellow]all time: [/yellow]{'{:.2f}'.format(best_fitness)}"
                                                         f"[green], curr gen:[/green] {'{:.2f}'.format(best_tree.fitness)}[grey39])[/grey39] "
                                                         f"[grey39](|T|:[/grey39] {int(best_tree.get_tree_size())}[grey39], [/grey39]"
                                                         f"[grey39]E[R]:[/grey39] {best_tree.reward:.2f} ± {best_tree.std_reward:.2f}[grey39], [/grey39]"
                                                         f"[grey39]SR:[/grey39] {(best_tree.success_rate):.2f}[grey39])[/grey39]")

    tok = time.perf_counter()
    best_tree.elapsed_time = tok - tik

    return best_tree, full_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cartpole")
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lamb", type=int, default=10)
    parser.add_argument("--mu", type=int, default=1)
    parser.add_argument("--n_gens", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--plateau_limit", type=int, default=None)
    parser.add_argument("--filename", type=str, default="es")
    parser.add_argument("--initial_pop", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--should_normalize_initial_pop", action="store_true")
    parser.add_argument("--penalize_std", action="store_true")
    args = parser.parse_args()

    config = get_config(args.task)
    cmd_line = str(args) + "\n" + "-" * 50 + "\n"
    filepath = "results/" + datetime.now().strftime(f"{args.filename}_%Y-%m-%d_%H-%M-%S")

    best_log = []
    full_log = []

    for simulation in range(args.simulations):
        initial_pop = load_initial_pop(args.initial_pop, simulation, args.should_normalize_initial_pop)

        dt, log = evolution_strategy(config, args.alpha, args.episodes, args.lamb, args.mu, args.n_gens, args.depth,
                                     initial_pop=initial_pop, simulation_id=simulation,
                                     plateau_limit=args.plateau_limit, penalize_std=args.penalize_std,
                                     n_jobs=args.n_jobs, verbose=args.verbose)

        print(f"Evaluating best individual from simulation {simulation}...")
        print(f"Before: {dt.reward:.2f} ± {dt.std_reward:.2f} (SR: {dt.success_rate:.2f})")
        fill_metrics(config, [dt], args.alpha, 1000, should_norm_state=True, penalize_std=args.penalize_std,
                     task_solution_threshold=config["task_solution_threshold"], n_jobs=args.n_jobs)
        print(f"After: {dt.reward:.2f} ± {dt.std_reward:.2f} (SR: {dt.success_rate:.2f})")

        best_log.append((dt, dt.reward, dt.std_reward, dt.get_tree_size(), dt.success_rate))
        full_log.append(log)

        save_history_to_file(config, best_log, filepath + ".txt", prefix=cmd_line)
        save_full_history_to_file(config, full_log, filepath + "_tmp.txt", prefix=cmd_line)
        print(f"Saved to '{filepath}'.")

        plot_full_log(config, full_log, filepath + ".png")
