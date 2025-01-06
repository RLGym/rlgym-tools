from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED


class WavedashReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, scale_by_acceleration: bool = True):
        self.scale_by_acceleration = scale_by_acceleration
        self.prev_state = None
        self.prev_acceleration = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.prev_acceleration = {agent: 0 for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            prev_car = self.prev_state.cars[agent]

            wavedash = ((car.on_ground and not prev_car.on_ground)
                        and (car.is_flipping or prev_car.is_flipping))

            if self.scale_by_acceleration:
                if car.is_flipping and not prev_car.is_flipping:
                    acc = np.linalg.norm(car.physics.linear_velocity - prev_car.physics.linear_velocity)
                    self.prev_acceleration[agent] = acc
                if wavedash:
                    acc = self.prev_acceleration[agent]
                    rewards[agent] = acc / CAR_MAX_SPEED
                    self.prev_acceleration[agent] = 0
                else:
                    rewards[agent] = 0
                    if not car.is_flipping:
                        self.prev_acceleration[agent] = 0
            else:
                if wavedash:
                    rewards[agent] = 1
                else:
                    rewards[agent] = 0
        self.prev_state = state

        return rewards
