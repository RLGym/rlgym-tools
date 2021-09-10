from collections import deque
from typing import Any

import numpy as np

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
        self.obs = obs
        self.queue = deque([], maxlen=self.stack_size)
        self.frame_n = 0
        self.new = True

    def reset(self, initial_state: GameState):
        self.new = True

    def build_obs(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> Any:
        # sort out the edge case after reset of environment
        if self.new:
            initial_obs = self.obs.build_obs(player, state, previous_action)
            for _ in range(self.stack_size):
                self.queue.appendleft(initial_obs)
        self.new = False

        frame = self.obs.build_obs(player, state, previous_action)

        self.frame_n += 1
        self.queue.appendleft(frame)
        return np.concatenate(self.queue)
