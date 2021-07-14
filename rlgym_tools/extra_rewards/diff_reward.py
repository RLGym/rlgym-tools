import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState


class DiffReward(RewardFunction):
    """
    Calculates difference in reward between time steps
    For instance, it can be used to reward acceleration by measuring change in velocity.
    """
    def __init__(self, reward_function: RewardFunction):
        super().__init__()
        self.reward_function = reward_function
        self.last_values = {}

    def reset(self, initial_state: GameState):
        self.last_values = {}

    def _calculate_diff(self, player, rew):
        last = self.last_values.get(player.car_id)
        self.last_values[player.car_id] = rew
        if last is not None:
            return rew - last
        else:
            return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        return self._calculate_diff(player, rew)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        return self._calculate_diff(player, rew)
