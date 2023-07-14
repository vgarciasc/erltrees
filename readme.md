# erltrees

This repository contains utilities for working with Decision Trees for Reinforcement Learning. These utilities include things like evaluating a tree, printing it as a string, visualizing it in real-time, as well as applying RL algorithms like DAgger and the newly-proposed Reward Pruning.

## How to install

The code was tested with Python 3.10. To install the dependencies, clone the repository, `cd` into the folder and run:

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Task configurations

To add a new task, go to `erltrees/rl/configs.py` and add a new configuration following this template:

```python
config_cp = {
    "name": "CartPole-v1",
    "can_render": True,
    "max_score": 500,
    "min_score": 0,
    "task_solution_threshold": 495,
    "action_type": "discrete",
    "n_actions": 2,
    "actions": ["left", "right"],
    "n_attributes": 4,              
    "attributes": [
        ("Cart Position", "continuous", [-4.8, 4.8]),
        ("Cart Velocity", "continuous", [-math.inf, math.inf]),
        ("Pole Angle", "continuous", [-0.418, 0.418]),
        ("Pole Angular Velocity", "continuous", [-math.inf, math.inf])],
    "maker": lambda : gym.make("CartPole-v1"),
}
```

Then, add the configuration to `get_config` with its corresponding ID. This allows to easily select the task to solve in any part of the code.

## Imitation Learning

Two different algorithms are provided for Imitation Learning: Behavioral Cloning and DAgger. The latter can be run by executing `python -m erltrees.rl.dagger`, and providing the following parameters:

- `--task`: the ID of the RL task to solve.
- `--class`: `ClassificationTree` or `RegressorTree`, depending on the type of tree to use.
- `--expert_class`: the class of the expert DNN. Can be `MLP`, `KerasDNN` or `PPO`, depending on how the expert was trained.
- `--expert_path`: the path to the expert DNN. If `expert_class` is `PPO`, this should be a folder containing the `model.zip` file.
- `--pruning`: the ccp_alpha value for pruning the tree. If 0, no pruning is applied.
- `--fitness_alpha`: the alpha value used when calculating fitness. If 0, tree size is not taken into account when selecting the best tree.
- `--iterations`: the number of iterations to run DAgger for.
- `--episodes`: the number of episodes to run for each iteration.
- `--should_collect_dataset`: if True, the dataset will be collected before running DAgger. If False, the dataset will be loaded from a file with the same name as the `expert_path`, but finishing with `_dataset`.
- `--dataset_size`: the size of the dataset to collect.
- `--simulations`: the number of simulations to run.
- `--should_save_models`: if True, the models will be saved after each iteration. If False, they will not be saved.
- `--should_only_save_best`: if True, only the best model will be saved. If False, all models will be saved.
- `--save_models_path`: the path where the models will be saved.
- `--n_jobs`: the number of jobs to run in parallel.
- `--verbose`: if True, the progress will be printed to the console.
- `--seed`: the seed to use for the random number generator.

Example:

```bash
python -m erltrees.il.dagger -t cartpole -c ClassificationTree -e PPO -f "models/experts/cartpole_PPO" -p 0.002 -a 0.000001 -i 50 -j 100 --should_save_models True --should_only_save_best True --save_models_path "debug" --verbose True --n_jobs 8 --simulations 50
```

Behavioral Cloning functions in a very similar way, and is located in `erltrees.rl.behavioral_cloning`. The parameters are the same, except for `fitness_alpha`, `iterations`, `episodes`, `should_collect_dataset` and `n_jobs`, which are not used.

## Reward Pruning

Reward Pruning, a method of reducing the size of Reinforcement Learning Decision Trees while maintaining their performance, is a new method proposed in "Evolving Interpretable Decision Trees for Reinforcement Learning". 
This method can be run by executing `python -m erltrees.rl.reward_pruning`, and providing the following parameters:

- `--task`: the ID of the RL task to solve.
- `--class`: `ClassificationTree` or `RegressorTree`, depending on the type of tree to use.
- `--alpha`: the alpha value used when calculating fitness. If 0, tree size is not taken into account when selecting the best tree.
- `--episodes`: the number of episodes to run when evaluating a tree.
- `--rounds`: the number of rounds to run Reward Pruning for.
- `--simulations`: the number of simulations to run.
- `--norm_state`: if True, the state will be normalized before being used to train the tree.
- `--should_penalize_std`: if True, the standard deviation of the rewards will be penalized when calculating fitness.
- `--n_jobs`: the number of jobs to run in parallel.

Example:

```bash
python -m erltrees.experiments.reward_pruning -t cartpole -f "../CRO_DT_RL/results/complete/cartpole_IL_ppo_p002.txt" -a 0.0001 --rounds 10 --simulations 50 --n_jobs 8 --norm_state False --episodes 1000 -o "../CRO_DT_RL/results/complete/cartpole_ppo_p002_reward_pruning.txt"
```

## Visualization

The code for visualizing a tree in real-time is located in `erltrees.viz.evaluate_agent`. To run it, execute `python -m erltrees.visualization --task [TASK_NAME]` where `[TASK_NAME]` is the ID of the task to solve.
Only `cartpole`, `mountain_car` and `lunar_lander` are supported at the moment, but it is easy to extend this code to other tasks.