import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from stable_baselines3.common.logger import Logger


class SB3LogReward(RewardFunction):
    """
    Simple reward function for logging individual rewards to a custom Logger.
    """
    def __init__(self, logger: Logger, reward_function: RewardFunction):
        super().__init__()
        self.logger = logger
        self.reward_function = reward_function
        self.reward_sum = 0
        self.episode_steps = 0
        self.global_steps = 0

    def reset(self, initial_state: GameState):
        if self.episode_steps > 0:
            rew_fn_type = type(self.reward_function)
            mean_reward = self.reward_sum / self.episode_steps
            if rew_fn_type.__str__ is not object.__str__:
                self.logger.record(f"{self.reward_function}/ep_rew_mean", mean_reward)
            else:
                self.logger.record(f"{rew_fn_type.__name__}/ep_rew_mean", mean_reward)
            self.logger.dump(self.global_steps)
            self.reward_sum = 0
            self.episode_steps = 0
            self.global_steps += 1

        self.reward_function.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        self.reward_sum += rew
        self.episode_steps += 1
        return rew

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        self.reward_sum += rew
        self.episode_steps += 1
        return rew
