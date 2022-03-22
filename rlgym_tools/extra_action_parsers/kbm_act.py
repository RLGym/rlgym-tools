from typing import Any

import gym
import numpy as np

from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.gamestates import GameState


class KBMAction(DiscreteAction):
    def get_action_space(self) -> gym.spaces.Space:
        # throttle/pitch, steer/pitch/roll, jump, boost and handbrake/air roll
        return gym.spaces.MultiDiscrete([self._n_bins] * 2 + [2] * 3)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 5))
        actions[:, 0] = actions[:, 0] / (self._n_bins // 2) - 1
        actions[:, 1] = actions[:, 1] / (self._n_bins // 2) - 1

        parsed = np.zeros((actions.shape[0], 8))
        parsed[:, 0] = actions[:, 0]  # throttle
        parsed[:, 1] = actions[:, 1]  # steer
        parsed[:, 2] = -actions[:, 0]  # pitch
        parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])  # yaw
        parsed[:, 4] = actions[:, 1] * actions[:, 4]  # roll
        parsed[:, 5] = actions[:, 2]  # jump
        parsed[:, 6] = actions[:, 3]  # boost
        parsed[:, 7] = actions[:, 4]  # handbrake

        return parsed
