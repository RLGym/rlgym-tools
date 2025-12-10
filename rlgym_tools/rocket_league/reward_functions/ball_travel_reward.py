from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import euclidean_distance


class BallTravelReward(RewardFunction[AgentID, GameState, float]):
    def __init__(
            self,
            self_to_self_weight=0.0,
            self_to_tm8_weight=0.0,
            tm8_to_self_weight=0.0,
            self_to_opp_weight=-0.0,
            opp_to_self_weight=0.0,
            self_to_opp_goal_weight=0.0,
            self_to_own_goal_weight=-0.0,
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
        assert (self_to_self_weight != 0
                or self_to_tm8_weight != 0
                or tm8_to_self_weight != 0
                or self_to_opp_weight != 0
                or opp_to_self_weight != 0
                or self_to_opp_goal_weight != 0
                or self_to_own_goal_weight != 0), "At least one weight must be non-zero."

        self.self_to_self_weight = self_to_self_weight
        self.self_to_tm8_weight = self_to_tm8_weight
        self.tm8_to_self_weight = tm8_to_self_weight
        self.self_to_opp_weight = self_to_opp_weight
        self.opp_to_self_weight = opp_to_self_weight
        self.self_to_opp_goal_weight = self_to_opp_goal_weight
        self.self_to_own_goal_weight = self_to_own_goal_weight

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
        self.last_touch_agents = None
        self.distance_since_touch = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_pos = initial_state.ball.position
        self.last_touch_agents = [agent for agent in agents if initial_state.cars[agent].ball_touches > 0]
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

        # This list is to remove dependence on agent order
        touching_agents = [agent for agent in state.cars if state.cars[agent].ball_touches > 0]

        # Assign rewards based on the ball touches
        rewards = {k: 0.0 for k in state.cars}
        for a1 in self.last_touch_agents:
            for a2 in touching_agents:
                if a1 == a2:
                    # Same agent touched the ball again
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    rewards[a1] += norm_dist * self.self_to_self_weight
                elif state.cars[a1].team_num == state.cars[a2].team_num:
                    # Teammate touch
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    rewards[a1] += norm_dist * self.self_to_tm8_weight
                    rewards[a2] += norm_dist * self.tm8_to_self_weight
                else:
                    # Opponent touch
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    rewards[a1] += norm_dist * self.self_to_opp_weight
                    rewards[a2] += norm_dist * self.opp_to_self_weight

        # Goal scored rewards
        if state.goal_scored:
            team = state.scoring_team
            norm_dist = self.distance_since_touch * self.normalization_factor
            for lta in self.last_touch_agents:
                if team == state.cars[lta].team_num:
                    rewards[lta] += norm_dist * self.self_to_opp_goal_weight
                else:
                    rewards[lta] += norm_dist * self.self_to_own_goal_weight

        if len(touching_agents) > 0:
            norm_factor = len(touching_agents) * len(self.last_touch_agents)

            if norm_factor > 1:
                # If multiple agents touch the ball in the same step, adjust rewards
                for agent in state.cars:
                    rewards[agent] /= norm_factor

            # Update the last touch agents
            self.last_touch_agents = touching_agents
            self.distance_since_touch = 0

        shared_info["last_touch_agents"] = self.last_touch_agents
        shared_info["distance_since_touch"] = self.distance_since_touch

        # Only return rewards for the requested agents
        rewards = {agent: rewards[agent] for agent in agents}

        return rewards
