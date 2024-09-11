from typing import Any, Dict, List

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class WavedashReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {
            agent: float(state.cars[agent].on_ground and self.prev_state[agent].is_flipping)
            for agent in agents
        }
        self.prev_state = state

        return rewards
