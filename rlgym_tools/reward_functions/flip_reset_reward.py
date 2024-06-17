from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class FlipResetReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.can_flip and not self.prev_state.cars[agent].can_flip:
                down = -car.physics.up
                car_ball = state.ball.position - car.physics.position
                car_ball = car_ball / np.linalg.norm(car_ball)
                cossim_down_ball = np.dot(down, car_ball)
                if cossim_down_ball > 0.5 ** 0.5:  # 45 degrees
                    rewards[agent] = 1
        self.prev_state = state
        return rewards
