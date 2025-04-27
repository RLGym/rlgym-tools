from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED


class DemoReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, attacker_reward: float = 1.0, victim_punishment: float = 1.0,
                 bump_acceleration_reward: float = 0.0):
        self.attacker_reward = attacker_reward
        self.victim_punishment = victim_punishment
        self.bump_acceleration_reward = bump_acceleration_reward

        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            victim = car.bump_victim_id
            if victim is not None:
                victim_car = state.cars[victim]
                if victim_car.is_demoed:
                    if not self.prev_state.cars[victim].is_demoed:
                        rewards[agent] += self.attacker_reward
                        rewards[victim] -= self.victim_punishment
                else:
                    acceleration = np.linalg.norm(state.cars[victim].physics.linear_velocity
                                                  - self.prev_state.cars[victim].physics.linear_velocity)
                    is_teammate = car.team_num == victim_car.team_num
                    reward = self.bump_acceleration_reward * acceleration / CAR_MAX_SPEED
                    rewards[agent] += reward if not is_teammate else -reward

        self.prev_state = state

        return rewards
