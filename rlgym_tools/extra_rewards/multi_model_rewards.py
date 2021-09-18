from typing import List

import numpy as np
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import DefaultReward


class MultiModelReward(DefaultReward):
    """
    Handles the distribution of rewards to specific models where each model uses a different reward function
    """
    index_tracker = 0

    def __init__(self, model_map: List[int], reward_funcs: List[DefaultReward]):
        """
        Handles the distribution of rewards to specific models where each model uses a different reward function

        :param model_map: A list containing the mapping of model index to player
        :param reward_funcs: A list of reward functions for each model, in the same order as the list
            of models used elsewhere
        """
        if max(model_map) >= len(reward_funcs):
            raise ValueError("model_map implies the existence of more models than reward funcs")

        super().__init__()
        self.model_map = model_map
        self.reward_funcs = reward_funcs
        # This will make sure the right instance index is passed
        self.index = self.index_tracker
        self.index_tracker += 1

    def reset(self, initial_state: GameState):
        for func in self.reward_funcs:
            func.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # look up which model it is
        model_num = self.model_map[
            self.index * len(state.players) + [bot.car_id for bot in state.players].index(player.car_id)]

        return self.reward_funcs[model_num].get_reward(player, state, previous_action)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # look up which model it is
        model_num = self.model_map[
            self.index * len(state.players) + [bot.car_id for bot in state.players].index(player.car_id)]

        return self.reward_funcs[model_num].get_final_reward(player, state, previous_action)