from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

RAMP_HEIGHT = 256


class AerialDistanceReward(RewardFunction[AgentID, GameState, float]):
    """
    Aerial distance reward.
    - First aerial touch is rewarded by height
    - Consecutive touches based on distance travelled (since last aerial touch)
    - Resets when grounded or when another player touches the ball
    """

    def __init__(
            self,
            touch_height_weight: float = 1.0,
            car_distance_weight: float = 1.0,
            ball_distance_weight: float = 1.0
    ):
        super().__init__()
        self.touch_height_weight = touch_height_weight
        self.car_distance_weight = car_distance_weight
        self.ball_distance_weight = ball_distance_weight
        self.distances = {}
        self.last_touch_agent = None
        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.distances = {k: 0 for k in agents}
        self.last_touch_agent = None
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            if self.last_touch_agent == agent:
                if car.physics.position[2] < RAMP_HEIGHT:
                    self.distances[agent] = 0
                    self.last_touch_agent = None
                else:
                    dist = (np.linalg.norm(car.physics.position - self.prev_state.cars[agent].physics.position)
                            + np.linalg.norm(state.ball.position - self.prev_state.ball.position))
                    self.distances[agent] += dist
            if car.ball_touches > 0:
                rewards[agent] = self.distances[agent]
                if self.last_touch_agent == agent:
                    rewards[agent] += self.distances[agent]
                    self.distances[agent] = 0
                else:
                    touch_height = float(car.physics.position[2] + state.ball.position[2]) / 2
                    touch_height = max(0, touch_height - RAMP_HEIGHT)  # Clamp to 0
                    rewards[agent] += touch_height * self.touch_height_weight
                    self.last_touch_agent = agent
        self.prev_state = state
        shared_info["aerial_distance_info"] = {"distances": self.distances, "last_touch_agent": self.last_touch_agent}
        return rewards
