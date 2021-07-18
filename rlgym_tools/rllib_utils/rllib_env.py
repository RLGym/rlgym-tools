from typing import Tuple

import numpy as np
import ray
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict

from rlgym.gym import Gym


def _action_dict_to_numpy(action_dict: MultiAgentDict):
    action_array = np.zeros((len(action_dict), 8))
    for i, act in action_dict.items():
        action_array[i][:] = act
    return action_array


class RLLibEnv(MultiAgentEnv):
    def __init__(self, env: Gym):
        self.env = env

    @override(MultiAgentEnv)
    def reset(self) -> MultiAgentDict:
        observations = self.env.reset()
        return {i: obs for i, obs in enumerate(observations)}

    @override(MultiAgentEnv)
    def step(self, action_dict: MultiAgentDict) -> \
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        action_array = _action_dict_to_numpy(action_dict)

        observations, rewards, done, info = self.env.step(action_array)

        obs_dict = {}
        rew_dict = {}
        done_dict = {"__all__": done}
        info_dict = {}
        for i, (obs, rew) in enumerate(zip(observations, rewards)):
            obs_dict[i] = obs
            rew_dict[i] = rew
            info_dict[i] = info
        return obs_dict, rew_dict, done_dict, info_dict
