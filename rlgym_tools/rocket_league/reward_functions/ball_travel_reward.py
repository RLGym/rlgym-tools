from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z


class BallTravelReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, consecutive_weight=1.0,
                 pass_weight=1.0, receive_weight=1.0,
                 giveaway_weight=-1.0, intercept_weight=1.0,
                 goal_weight=1.0,
                 distance_normalization=None,
                 do_integral=False):
        """
        Reward function based on the distance the ball travels between touches.

        :param consecutive_weight: Weight for distance covered between consecutive touches by the same player.
        :param pass_weight: Weight for distance covered by a pass to a teammate.
        :param receive_weight: Weight for distance covered by a pass received from a teammate.
        :param giveaway_weight: Weight for distance covered by a pass (giveaway) to an opponent.
        :param intercept_weight: Weight for distance covered by a pass intercepted from an opponent.
        :param goal_weight: Weight for distance covered between a touch and a goal.
        :param distance_normalization: Factor to normalize distance travelled between touches.
                                       Defaults to weighting a distance of the full length of the field as 1.0
        :param do_integral: Whether to calculate the area under the ball's travel curve instead of the distance.
        """
        self.consecutive_weight = consecutive_weight
        self.pass_weight = pass_weight
        self.receive_weight = receive_weight
        self.giveaway_weight = giveaway_weight
        self.intercept_weight = intercept_weight
        self.goal_weight = goal_weight

        if distance_normalization is None:
            if do_integral:
                # Use the area of half a field length by half ceiling height
                distance_normalization = 4 / (2 * BACK_WALL_Y * CEILING_Z)
            else:
                # Use the full length of the field
                distance_normalization = 1 / (2 * BACK_WALL_Y)
        self.normalization_factor = distance_normalization
        self.do_integral = do_integral

        self.prev_ball_pos = None
        self.last_touch_agent = None
        self.distance_since_touch = 0

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_pos = initial_state.ball.position
        self.last_touch_agent = None
        self.distance_since_touch = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position

        # Update the distance travelled by the ball
        distance = np.linalg.norm(ball_pos - self.prev_ball_pos)
        if self.do_integral:
            # The path of the ball defines a right trapezoid (to a close approximation).
            z_height = (ball_pos[2] + self.prev_ball_pos[2]) / 2
            area = distance * z_height
            distance = area
        self.prev_ball_pos = ball_pos
        self.distance_since_touch += distance

        # Assign rewards based on the ball touches
        rewards = {k: 0.0 for k in agents}
        touching_agents = []  # This list is to remove dependence on agent order
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                if self.last_touch_agent is not None:
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    if agent == self.last_touch_agent:
                        # Consecutive touches
                        rewards[agent] += norm_dist * self.consecutive_weight
                    elif car.team_num == state.cars[self.last_touch_agent].team_num:
                        # Pass to teammate
                        rewards[agent] += norm_dist * self.receive_weight
                        rewards[self.last_touch_agent] += norm_dist * self.pass_weight
                    else:
                        # Team change
                        rewards[agent] += norm_dist * self.intercept_weight
                        rewards[self.last_touch_agent] += norm_dist * self.giveaway_weight
                touching_agents.append(agent)
            elif car.is_demoed and self.last_touch_agent == agent:
                self.last_touch_agent = None

        if state.goal_scored and self.last_touch_agent is not None:
            team = state.scoring_team
            norm_dist = self.distance_since_touch * self.normalization_factor
            mul = 1 if team == state.cars[self.last_touch_agent].team_num else -1
            rewards[self.last_touch_agent] += mul * norm_dist * self.goal_weight

        if len(touching_agents) > 0:
            self.distance_since_touch = 0
            # Update the last touch agent
            if len(touching_agents) == 1:
                self.last_touch_agent = touching_agents[0]
            else:
                # If multiple agents touch the ball in the same step, adjust rewards
                for agent in agents:
                    rewards[agent] /= len(touching_agents)
                # and set last touch to be the one that is closest to the ball
                closest_agent = min(touching_agents,
                                    key=lambda x: np.linalg.norm(state.cars[x].physics.position - ball_pos))
                self.last_touch_agent = closest_agent

        shared_info["last_touch_agent"] = self.last_touch_agent
        shared_info["distance_since_touch"] = self.distance_since_touch

        return rewards
