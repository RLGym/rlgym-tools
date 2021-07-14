import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState


class MultiplyRewards(RewardFunction):
    def __init__(self, *reward_functions: RewardFunction):
        super().__init__()
        self.reward_functions = reward_functions

    def reset(self, initial_state: GameState):
        for rew_func in self.reward_functions:
            rew_func.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.product([rf.get_reward(player, state, previous_action) for rf in self.reward_functions])

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.product([rf.get_final_reward(player, state, previous_action) for rf in self.reward_functions])
