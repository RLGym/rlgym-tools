from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState


class BaseWrapper(RewardFunction[AgentID, GameState, float]):
    """
    A convenience class to gather all wrappers under a same parent
    """
    def __init__(self, reward_function: RewardFunction[AgentID, GameState, float]):
        self.reward_function = reward_function

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return self.reward_function.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_function.reset(agents, initial_state, shared_info)