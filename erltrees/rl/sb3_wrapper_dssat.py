import gym
import numpy as np
import pdb


class Formator():
    def __init__(self, env):
        self.action_space_dict = env.action_space
        self.action_names = [*env.action_space]
        self.observation_dict_to_array = env.observation_dict_to_array

    def _get_action_bounds(self, action_name):
        return (self.action_space_dict[action_name].low, self.action_space_dict[action_name].high)

    def _check_array_actions(self, actions):
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        return actions

    # helpers for action normalization
    def normalize_actions(self, actions):
        """Normalize the action from [low, high] to [-1, 1]"""
        actions = self._check_array_actions(actions)
        normalized_actions = []
        for action_name, action in zip(self.action_names, actions):
            low, high = self._get_action_bounds(action_name)
            normalized_action = 2.0 * ((action - low) / (high - low)) - 1.0
            normalized_actions.append(normalized_action)
        return normalized_actions

    def denormalize_actions(self, actions):
        """Denormalize the action from [-1, 1] to [low, high]"""
        actions = self._check_array_actions(actions)
        denormalized_actions = []
        for action_name, action in zip(self.action_names, actions):
            low, high = self._get_action_bounds(action_name)
            denormalized_action = low + (0.5 * (action + 1.0) * (high - low))
            denormalized_actions.append(denormalized_action)
        return denormalized_actions

    def format_actions(self, actions):
        actions = self._check_array_actions(actions)
        return {action_name: action for action_name, action in zip([*self.action_space_dict], actions)}

    def format_observation(self, observation):
        return self.observation_dict_to_array(observation)


class GymDssatWrapper(gym.Wrapper):
    """
    Wrapper for easy and uniform interfacing with SB3
    """

    def __init__(self, env):
        self.formator = Formator(env)
        super().__init__(env)
        # using a normalized action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.formator.action_names),),
                                           dtype="float32")

        # using a vector representation of observations to allow
        # easily using SB3 MlpPolicy
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=np.inf,
                                                shape=env.observation_dict_to_array(
                                                    env.observation).shape,
                                                dtype="float32"
                                                )

        # to avoid annoying problem with Monitor when episodes end and things are None
        self.last_info = {}
        self.last_obs = None

    def reset(self):
        return self.formator.format_observation(self.env.reset())

    def step(self, action):
        # Rescale action from [-1, 1] to original action space interval
        denormalized_action = self.formator.denormalize_actions(action)
        formatted_action = self.formator.format_actions(denormalized_action)
        obs, reward, done, info = self.env.step(formatted_action)

        # handle `None` in obs, reward, and info on done step
        if done:
            obs, reward, info = self.last_obs, 0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self.formator.format_observation(obs)
        return formatted_observation, reward, done, info

    def close(self):
        return self.env.close()

    def eval(self):
        return self.env.set_evaluation()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    pass
