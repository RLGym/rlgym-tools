from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y

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
            ball_distance_weight: float = 1.0,
            distance_normalization: float = 1 / BACK_WALL_Y
    ):
        super().__init__()
        self.touch_height_weight = touch_height_weight
        self.car_distance_weight = car_distance_weight
        self.ball_distance_weight = ball_distance_weight
        self.distance_normalization = distance_normalization
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
                    dist_car = np.linalg.norm(car.physics.position - self.prev_state.cars[agent].physics.position)
                    dist_ball = np.linalg.norm(state.ball.position - self.prev_state.ball.position)
                    self.distances[agent] += (dist_car * self.car_distance_weight
                                              + dist_ball * self.ball_distance_weight)
            if car.ball_touches > 0:
                if self.last_touch_agent == agent:
                    norm_dist = self.distances[agent] * self.distance_normalization
                    rewards[agent] += norm_dist
                else:
                    w1 = self.car_distance_weight
                    w2 = self.ball_distance_weight
                    if w1 == w2 == 0:
                        w1 = w2 = 1
                    touch_height = float((w1 * car.physics.position[2] + w2 * state.ball.position[2]) / (w1 + w2))
                    touch_height = max(0.0, touch_height - RAMP_HEIGHT)  # Clamp to 0
                    norm_dist = touch_height * self.distance_normalization
                    rewards[agent] += norm_dist * self.touch_height_weight
                    self.last_touch_agent = agent
                self.distances[agent] = 0
        self.prev_state = state
        shared_info["aerial_distance_info"] = {"distances": self.distances, "last_touch_agent": self.last_touch_agent}
        return rewards
