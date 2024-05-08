from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CEILING_Z, BALL_RADIUS, BLUE_TEAM

import numpy as np


class TeamSpiritRewardWrapper(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward_fn: RewardFunction, team_spirit: float):
        self.reward_fn = reward_fn
        self.team_spirit = team_spirit

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        base_rewards = self.reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        total_blue = total_orange = n_blue = n_orange = 0
        for agent in agents:
            if state.cars[agent].team_num == BLUE_TEAM:
                total_blue += base_rewards[agent]
                n_blue += 1
            else:
                total_orange += base_rewards[agent]
                n_orange += 1

        mean_blue = total_blue / n_blue if n_blue > 0 else 0
        mean_orange = total_orange / n_orange if n_orange > 0 else 0

        spirit_rewards = {}
        for agent in agents:
            rew = base_rewards[agent]
            if state.cars[agent].team_num == BLUE_TEAM:
                team_mean, opp_mean = mean_blue, mean_orange
            else:
                team_mean, opp_mean = mean_orange, mean_blue
            spirit_rewards[agent] = (1 - self.team_spirit) * rew + self.team_spirit * team_mean - opp_mean

        return spirit_rewards
