from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState


class EpisodeEndReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward: float = 1.0):
        self.reward = reward

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        return {
            a: self.reward if is_terminated[a] else 0.0
            for a in agents
        }
