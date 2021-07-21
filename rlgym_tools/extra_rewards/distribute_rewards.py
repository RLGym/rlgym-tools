import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData


class DistributeRewards(RewardFunction):
    """
    Inspired by OpenAI's Dota bot (OpenAI Five).
    Modifies rewards using the formula (1-team_spirit) * own_reward + team_spirit * avg_team_reward - avg_opp_reward
        For instance, in a 3v3 where scoring a goal gives 100 reward with team_spirit 0.3 / 0.6 / 0.9:
            - Goal scorer gets 80 / 60 / 40
            - Teammates get 10 / 20 / 30 each
            - Opponents get -33.3 each

    Note that this will bring mean reward close to zero, so tracking might be misleading.
    If using one of the SB3 envs SB3DistributeRewardsWrapper can be used after logging.
    """
    def __init__(self, reward_func: RewardFunction, team_spirit=0.3):
        super().__init__()
        self.reward_func = reward_func
        self.team_spirit = team_spirit
        self.last_state = None
        self.base_rewards = {}
        self.avg_blue = 0
        self.avg_orange = 0

    def _compute(self, state: GameState, final=False):
        if state != self.last_state:
            self.base_rewards = {}
            sum_blue = 0
            n_blue = 0
            sum_orange = 0
            n_orange = 0
            for player in state.players:
                if final:
                    rew = self.reward_func.get_final_reward(player, state, None)
                else:
                    rew = self.reward_func.get_reward(player, state, None)

                self.base_rewards[player.car_id] = rew
                if player.team_num == BLUE_TEAM:
                    sum_blue += rew
                    n_blue += 1
                else:
                    sum_orange += rew
                    n_orange += 1
            self.avg_blue = sum_blue / (n_blue or 1)
            self.avg_orange = sum_orange / (n_orange or 1)

            self.last_state = state

    def _get_individual_reward(self, player):
        base_reward = self.base_rewards[player.car_id]
        if player.team_num == BLUE_TEAM:
            reward = self.team_spirit * self.avg_blue + (1 - self.team_spirit) * base_reward - self.avg_orange
        else:
            reward = self.team_spirit * self.avg_orange + (1 - self.team_spirit) * base_reward - self.avg_blue
        return reward

    def reset(self, initial_state: GameState):
        self.reward_func.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._compute(state, final=False)
        return self._get_individual_reward(player)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._compute(state, final=True)
        return self._get_individual_reward(player)
