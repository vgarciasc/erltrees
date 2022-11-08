import pdb
import numpy as np
import matplotlib.pyplot as plt

from erltrees.il.ann import MLPAgent
from erltrees.il.utils import get_dataset_from_model, load_dataset, save_dataset
from erltrees.il.keras_dnn import KerasDNN
import erltrees.rl.utils as rl

def handle_args(args, config):
    filename = args['expert_filepath']

    if args['expert_class'] == "KerasDNN":
        expert = KerasDNN(config, exploration_rate=args['expert_exploration_rate'])
        expert.load_model(filename)
    elif args['expert_class'] == "MLP":
        expert = MLPAgent(config, exploration_rate=args['expert_exploration_rate'])
        expert.load_model(filename)

    if args['should_collect_dataset']:
        X, y = get_dataset_from_model(config, expert, args['dataset_size'], args['verbose'])
        save_dataset(f"{filename}_dataset", X, y)
        print(f"Create dataset with {len(y)} observations for {config['name']}.")
    else:
        X, y = load_dataset(f"{filename}_dataset")
        print(f"Dataset length: {len(y)}")
        
    return expert, X, y