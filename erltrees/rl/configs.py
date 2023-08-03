import math
import gym
import numpy as np
# import gym_dssat_pdi

from erltrees.rl.sb3_wrapper_dssat import GymDssatWrapper

config_CP = {
    "name": "CartPole-v1",
    "can_render": True,
    "max_score": 500,
    "min_score": 0,
    "task_solution_threshold": 495,
    "should_force_episode_termination_score": True,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
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

config_MC = {
    "name": "MountainCar-v0",
    "can_render": True,
    "max_score": 0,
    "min_score": -200,
    "task_solution_threshold": -110,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
    "action_type": "discrete",
    "n_actions": 3,
    "actions": ["left", "nop", "right"],
    "n_attributes": 2,              
    "attributes": [
        ("Car Position", "continuous", [-1.2, 0.6]),
        ("Car Velocity", "continuous", [-0.07, 0.07])],
    "maker": lambda : gym.make("MountainCar-v0"),
}

config_LL = {
    "name": "LunarLander-v2",
    "can_render": True,
    "n_actions": 4,
    "max_score": 1000,
    "min_score": -10000,
    "task_solution_threshold": 200,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "episode_termination_score": 0,
    "actions": ["nop", "left engine", "main engine", "right engine"],
    "action_type": "discrete",
    "n_attributes": 8,              
    "attributes": [
        ("X Position", "continuous", [-1.5, 1.5]),
        ("Y Position", "continuous", [-1.5, 1.5]),
        ("X Velocity", "continuous", [-5.0, 5.0]),
        ("Y Velocity", "continuous", [-5.0, 5.0]),
        ("Angle", "continuous", [-math.pi, math.pi]),
        ("Angular Velocity", "continuous", [-5.0, 5.0]),
        ("Leg 1 is Touching", "binary", [0, 1]),
        ("Leg 2 is Touching", "binary", [0, 1])],
    "maker": lambda : gym.make("LunarLander-v2"),
}

config_BJ = {
    "name": "Blackjack-v0",
    "can_render": False,
    "max_score": 1,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": None,
    "action_type": "discrete",
    "n_actions": 2,
    "actions": ["stick", "hit"],
    "n_attributes": 3,
    "attributes": [
        ("Player's Sum", "discrete", 0, 22),
        ("Dealer's Card", "discrete", 1, 11),
        ("Usable Ace", "binary", -1, -1)],
    "maker": lambda : gym.make("Blackjack-v0"),
}

config_AB = {
    "name": "Acrobot-v1",
    "can_render": True,
    "max_score": 0,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": None,
    "action_type": "discrete",
    "n_actions": 3,
    "actions": ["minus_torque", "nop", "plus_torque"],
    "n_attributes": 6,
    "attributes": [
        ("Cosine Theta1", "continuous", (-1, +1)),
        ("Sine Theta1", "continuous", (-1, +1)),
        ("Cosine Theta2", "continuous", (-1, +1)),
        ("Sine Theta2", "continuous", (-1, +1)),
        ("Angvel Theta1", "continuous", (-12.567, 12.567)),
        ("Angvel Theta2", "continuous", (-28.274, 28.274))],
    "maker": lambda : gym.make("Acrobot-v1"),
}

config_DSSAT = {
    "name": "GymDssatPdi-v0",
    "can_render": False,
    "max_score": 0,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda a,b,c : c,
    "task_solution_threshold": 50,
    "episode_termination_score": None,
    "action_type": "continuous",
    "n_actions": -1,
    "actions": ["nitrogen"],
    "n_attributes": 11,
    "attributes": [
        ("cumsumfert", "continuous", (0, 50000)), # cumulative nitrogen fertilizer applications (kg/ha)
        ("dap", "continuous", (0, 366)),          # days after planting (day)
        ("dtt", "continuous", (0, 100)),          # growing degree days for current day (◦C/day)
        ("ep", "continuous", (0, 50)),            # actual plant transpiration rate (L/m2/day)
        ("grnwt", "continuous", (0, 50000)),      # grain weight dry matter (kg/ha)
        ("istage", "continuous", (0, 9)),         # DSSAT maize growing stage
        ("nstres", "continuous", (0, 1)),         # index of plant nitrogen stress (unitless)
        ("swfac", "continuous", (0, 1)),          # index of plant water stress (unitless)
        ("topwt", "continuous", (0, 1)),          # above the ground population biomass (kg/ha)
        ("vstage", "continuous", (0, 30)),        # vegetative growth stage (number of leaves)
        ("xlai", "continuous", (0, 10)),          # plant population leaf area index (m2 leaf/m2 soil)
    ],
    "maker": lambda : GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **{'mode': 'fertilization'})),
}

def get_config(task_name):
    if task_name == "cartpole":
        return config_CP
    elif task_name == "mountain_car":
        return config_MC
    elif task_name == "lunar_lander":
        return config_LL
    elif task_name == "blackjack":
        return config_BJ
    elif task_name == "acrobot":
        return config_AB
    elif task_name == "dssat":
        return config_DSSAT
        
    print(f"Invalid task_name {task_name}.")
    return None