import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from collections import deque


class GeneralStacker(ObsBuilder):
    '''
    Stacks several observations into one

    :param obs: Observation that is to be stacked.
    :param stack_size: How many past observations does the queue hold.

    '''

    def __init__(self, obs:ObsBuilder, stack_size: int = 15):
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
            initial_obs = self.obs.build_obs(player,state,previous_action)
            for _ in range(self.stack_size):
                self.queue.appendleft(initial_obs)
        self.new = False

        frame = self.obs.build_obs(player,state,previous_action)

        self.frame_n += 1
        self.queue.appendleft(frame)
        return np.concatenate(self.queue)
