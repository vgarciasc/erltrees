# Baseline agents for comparison
from sb3_wrapper import Formator
import numpy as np
import pdb
from erltrees.evo.evo_tree import Individual
from erltrees.rl.configs import get_config
from erltrees.rl.utils import normalize_state

class NullAgent:
    """
    Agent always choosing to do no fertilization
    """

    def __init__(self, env):
        self.env = env
        self.action_formator = Formator(env.env.env)

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        normalized_action = self.action_formator.normalize_actions([0])
        return np.array(normalized_action, dtype=np.float32), obs


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """

    def __init__(self, env):
        self.env = env
        self.action_formator = Formator(env.env.env)
        assert 'dap' in env.observation_variables
        self.dap_index = env.observation_variables.index('dap')
        all_policy_dic = {
            'fertilization': {
                40: 27,
                45: 35,
                80: 54,
            },
            'irrigation': {
                6: 13,
                20: 10,
                37: 10,
                50: 13,
                54: 18,
                65: 25,
                69: 25,
                72: 13,
                75: 15,
                77: 19,
                80: 20,
                84: 20,
                91: 15,
                101: 19,
                104: 4,
                105: 25,
            }
        }
        self.policy_dic = all_policy_dic[self.env.mode]

    def _policy(self, obs):
        obs = np.concatenate(obs, axis=None)
        dap = int(obs[self.dap_index])
        action = [self.policy_dic[dap] if dap in self.policy_dic else 0]
        return action

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        normalized_action = self.action_formator.normalize_actions(action)
        return np.array(normalized_action, dtype=np.float32), obs


class TreeAgent:
    """
    Agent always choosing to do no fertilization
    """

    def __init__(self, env, tree_str=None, denormalize=False):
        self.env = env
        self.action_formator = Formator(env.env.env)
        self.config = get_config("dssat")

        self.tree = None
        if tree_str is not None:
            self.tree = Individual.read_from_string(self.config, tree_str)
        if denormalize:
            self.tree.denormalize_thresholds()

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        # obs = normalize_state(self.config, obs)
        action = self.tree.act(obs)
        return np.array([action], dtype=np.float32), obs
        # normalized_action = self.action_formator.normalize_actions(action)
        # return np.array(normalized_action, dtype=np.float32), obs
