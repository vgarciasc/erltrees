import pdb
import argparse

from statsmodels.stats.proportion import proportion_confint
from functools import reduce
from collections import Counter
from rich import print

import erltrees.rl.utils as rl
import erltrees.il.utils as il
# from erltrees.il.ova import CartOvaAgent
from erltrees.rl.configs import get_config
from erltrees.il.ann import MLPAgent
from erltrees.il.keras_dnn import KerasDNN

def get_model(model_class, filename, config, expert=None):
    if model_class == "MLP":
        model = MLPAgent(config, 0.0)
        model.load_model(filename)
    elif model_class == "KerasDNN":
        model = KerasDNN(config, 0.0)
        model.load_model(filename)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rulelists')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-c','--imitator', help='Filepath for imitator', required=True)
    parser.add_argument('--imitator_class', help='Which type of file for the imitator?', required=True)
    parser.add_argument('-e','--expert', help='Filepath for expert', required=False)
    parser.add_argument('--expert_class', help='Which type of file for the expert?', required=False)
    parser.add_argument('-o','--output', help='Filepath to output converted tree', required=False)
    parser.add_argument('--dataset_size', help='Number of episodes used during pruning.', required=False, default=10, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # Initialization
    config = get_config(args['task'])
    print(f"[yellow]Creating dataset for {config['name']} with {args['dataset_size']} episodes.")
    print("")

    # Loading imitator
    imitator = get_model(
        model_class=args['imitator_class'],
        filename=args['imitator'],
        config=config)
    
    # Creating dataset
    X, y = il.get_dataset_from_model(
        config, imitator, 
        episodes=args['dataset_size'],
        verbose=args['verbose'])
    
    # Labeling dataset with expert...
    if args['expert']:
        print(f"[yellow]Labeling dataset with expert '{args['expert']}'...[/yellow]")
        expert = get_model(model_class=args['expert_class'], filename=args['expert'], config=config)
        y = il.label_dataset_with_model(config=config, model=expert, X=X)
    
    # Saving dataset
    il.save_dataset(args['output'], X, y)
    print(f"[yellow]Saved dataset to '{args['output']}'.[/yellow]")