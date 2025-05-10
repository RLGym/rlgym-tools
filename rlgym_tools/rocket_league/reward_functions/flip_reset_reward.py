from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.math import cosine_similarity


class FlipResetReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, obtain_flip_weight: float = 1.0, hit_ball_weight: float = 1.0):
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0.0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0 and car.has_flip and not self.prev_state.cars[agent].has_flip:
                down = -car.physics.up
                car_ball = state.ball.position - car.physics.position
                cossim_down_ball = cosine_similarity(down, car_ball)
                if cossim_down_ball > 0.5 ** 0.5:  # 45 degrees
                    self.has_reset.add(agent)
                    rewards[agent] = self.obtain_flip_weight
            elif car.on_ground:
                self.has_reset.discard(agent)
                self.has_flipped.discard(agent)
            elif car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)
            if car.ball_touches > 0 and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                rewards[agent] = self.hit_ball_weight
        self.prev_state = state
        return rewards
