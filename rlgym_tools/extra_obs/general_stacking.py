from typing import Any

import numpy as np

from rlgym.utils.common_values import NUM_ACTIONS
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder


class GeneralStacker(ObsBuilder):
    """
    Stacks several observations into one

    :param obs: observation that is to be stacked.
    :param stack_size: how many past observations the queue hold.

    """

    def __init__(self, obs: ObsBuilder, stack_size: int = 15):
        super().__init__()
        self.stack_size = stack_size
        self.obs_builder = obs

    def reset(self, initial_state: GameState):
        self.obs = {}
        previous_action = np.zeros(NUM_ACTIONS)
        for player in initial_state.players:
            self.obs_builder.reset(initial_state=initial_state)
            initial_obs = self.obs_builder.build_obs(player, initial_state, previous_action=previous_action)
            self.obs[player.car_id] = np.concatenate([initial_obs] * self.stack_size)
        self.obs_builder.reset(initial_state=initial_state)

    def build_obs(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> Any:
        frame = self.obs_builder.build_obs(player, state, previous_action)

        num = frame.shape[0]
        obs = self.obs[player.car_id]
        obs[num:] = obs[:-num]
        obs[:num] = frame

        return obs

