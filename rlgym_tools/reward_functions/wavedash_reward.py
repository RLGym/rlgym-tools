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
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            prev_car = self.prev_state.cars[agent]
            wavedash = ((car.on_ground and not prev_car.on_ground)
                        and (car.is_flipping or prev_car.is_flipping))
            if wavedash:
                rewards[agent] = 1
            else:
                rewards[agent] = 0
        self.prev_state = state

        return rewards
