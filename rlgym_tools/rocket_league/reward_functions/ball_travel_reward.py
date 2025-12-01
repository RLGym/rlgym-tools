from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import euclidean_distance


class BallTravelReward(RewardFunction[AgentID, GameState, float]):
    def __init__(
            self,
            self_to_self_weight=1.0,
            self_to_tm8_weight=1.0,
            tm8_to_self_weight=1.0,
            self_to_opp_weight=-1.0,
            opp_to_self_weight=1.0,
            self_to_opp_goal_weight=1.0,
            self_to_own_goal_weight=-1.0,
            invalidate_last_touch_on_demo=True,
            do_integral=False,
            distance_normalization=None,
    ):
        """
        Reward function based on the distance the ball travels between touches.

        :param self_to_self_weight: Weight for distance covered between consecutive touches by the same player.
        :param self_to_tm8_weight: Weight for distance covered by a pass to a teammate.
        :param tm8_to_self_weight: Weight for distance covered by a pass received from a teammate.
        :param self_to_opp_weight: Weight for distance covered by a pass (giveaway) to an opponent.
        :param opp_to_self_weight: Weight for distance covered by a pass intercepted from an opponent.
        :param self_to_opp_goal_weight: Weight for distance covered between a touch and a goal.
        :param distance_normalization: Factor to normalize distance travelled between touches.
                                       Defaults to weighting a distance of the full length of the field as 1.0
                                       or area of half field length by half ceiling height if do_integral is True.
        :param do_integral: Whether to calculate the area under the ball's travel curve instead of the distance.
        """
        self.self_to_self_weight = self_to_self_weight
        self.self_to_tm8_weight = self_to_tm8_weight
        self.tm8_to_self_weight = tm8_to_self_weight
        self.self_to_opp_weight = self_to_opp_weight
        self.opp_to_self_weight = opp_to_self_weight
        self.self_to_opp_goal_weight = self_to_opp_goal_weight
        self.self_to_own_goal_weight = self_to_own_goal_weight
        self.invalidate_last_touch_on_demo = invalidate_last_touch_on_demo

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

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_pos = initial_state.ball.position
        self.last_touch_agent = None
        for agent in initial_state.cars:
            if initial_state.cars[agent].ball_touches > 0:
                if self.last_touch_agent is not None:
                    ad = euclidean_distance(
                        initial_state.cars[agent].physics.position,
                        initial_state.ball.position
                    )
                    ltd = euclidean_distance(
                        initial_state.cars[self.last_touch_agent].physics.position,
                        initial_state.ball.position
                    )
                    if ltd < ad:
                        continue
                self.last_touch_agent = agent
        self.distance_since_touch = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position

        # Update the distance travelled by the ball
        distance = euclidean_distance(ball_pos, self.prev_ball_pos)
        if self.do_integral:
            # The path of the ball defines a right trapezoid (to a close approximation).
            z_height = (ball_pos[2] + self.prev_ball_pos[2]) / 2
            area = distance * z_height
            distance = area
        self.prev_ball_pos = ball_pos
        self.distance_since_touch += distance

        # Assign rewards based on the ball touches
        rewards = {k: 0.0 for k in state.cars}
        touching_agents = []  # This list is to remove dependence on agent order
        for agent in state.cars:
            car = state.cars[agent]
            if car.ball_touches > 0:
                if self.last_touch_agent is not None:
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    if agent == self.last_touch_agent:
                        # Consecutive touches
                        rewards[agent] += norm_dist * self.self_to_self_weight
                    elif car.team_num == state.cars[self.last_touch_agent].team_num:
                        # Pass to teammate
                        rewards[agent] += norm_dist * self.tm8_to_self_weight
                        rewards[self.last_touch_agent] += norm_dist * self.self_to_tm8_weight
                    else:
                        # Team change
                        rewards[agent] += norm_dist * self.opp_to_self_weight
                        rewards[self.last_touch_agent] += norm_dist * self.self_to_opp_weight
                touching_agents.append(agent)
            elif car.is_demoed and self.last_touch_agent == agent and self.invalidate_last_touch_on_demo:
                self.last_touch_agent = None

        if state.goal_scored and self.last_touch_agent is not None:
            team = state.scoring_team
            norm_dist = self.distance_since_touch * self.normalization_factor
            if team == state.cars[self.last_touch_agent].team_num:
                rewards[self.last_touch_agent] += norm_dist * self.self_to_opp_goal_weight
            else:
                rewards[self.last_touch_agent] += norm_dist * self.self_to_own_goal_weight

        if len(touching_agents) > 0:
            self.distance_since_touch = 0
            # Update the last touch agent
            if len(touching_agents) == 1:
                self.last_touch_agent = touching_agents[0]
            else:
                # If multiple agents touch the ball in the same step, adjust rewards
                for agent in state.cars:
                    rewards[agent] /= len(touching_agents)
                # and set last touch to be the one that is closest to the ball
                closest_agent = min(touching_agents,
                                    key=lambda x: euclidean_distance(state.cars[x].physics.position, ball_pos))
                self.last_touch_agent = closest_agent

        shared_info["last_touch_agent"] = self.last_touch_agent
        shared_info["distance_since_touch"] = self.distance_since_touch

        # Only return rewards for the requested agents
        rewards = {agent: rewards[agent] for agent in agents}

        return rewards
