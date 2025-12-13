from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.reward_functions.wrappers.base_wrapper import BaseWrapper


class WeightedWrapper(BaseWrapper):
    """
    Weights a reward based on the following formula:

    reward * weight
    """
    def __init__(self, reward_function: RewardFunction[AgentID, GameState, float], weight: float):
        super().__init__(reward_function)
        self.weight = weight

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = super().get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        for agent in agents:
            rewards[agent] *= self.weight

        return rewards