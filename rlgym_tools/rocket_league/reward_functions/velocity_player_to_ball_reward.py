from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, \
    BALL_RADIUS


class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, include_negative_values: bool = True, use_trajectory_comparison: bool = False,
                 use_dot_quotient: bool = False):
        self.include_negative_values = include_negative_values
        self.use_trajectory_comparison = use_trajectory_comparison
        self.use_dot_quotient = use_dot_quotient

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics
        if self.use_trajectory_comparison:
            curr_dist, min_dist, t = trajectory_comparison(car.position, car.linear_velocity,
                                                           ball.position, ball.linear_velocity)
            vel = (curr_dist - min_dist) / t if t != 0 else 0
            norm_vel = vel / (CAR_MAX_SPEED + BALL_MAX_SPEED)
            if abs(norm_vel) > 1:  # In case of floating point errors with small t
                norm_vel = np.sign(norm_vel)
        elif self.use_dot_quotient:
            car_to_ball = ball.position - car.position
            car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d which becomes v . d / |d|^2
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            vd = np.dot(car_to_ball, car.linear_velocity)
            dd = np.dot(car_to_ball, ball.linear_velocity)
            inv_time = vd / dd if dd != 0 else 0
            norm_vel = inv_time / (CAR_MAX_SPEED / BALL_RADIUS)
        else:
            car_to_ball = ball.position - car.position
            car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

            vel = np.dot(car_to_ball, car.linear_velocity)
            norm_vel = vel / CAR_MAX_SPEED
        if self.include_negative_values:
            return norm_vel
        return max(0, norm_vel)


def trajectory_comparison(pos1, vel1, pos2, vel2, check_bounds=True):
    """
    Calculate the closest point between two trajectories, defined as the lines:
      pos1 + t * vel1
      pos2 + t * vel2
    """
    # First, find max time based on field bounds
    if check_bounds:
        max_time = np.inf
        for pos, vel in (pos1, vel1), (pos2, vel2):
            bounds = np.array([[-SIDE_WALL_X, -BACK_WALL_Y, 0],
                               [SIDE_WALL_X, BACK_WALL_Y, CEILING_Z]])
            times = (bounds - pos) / (vel + (vel == 0))
            times = times[times > 0]
            t = np.min(times)
            max_time = min(max_time, t)

    # The distance between the two rays is `||pos1 + t * vel1 - pos2 - t * vel2||`
    # This is equivalent to `||(pos1 - pos2) + t * (vel1 - vel2)||`
    pos_diff = pos1 - pos2
    vel_diff = vel1 - vel2

    # The minimum distance is achieved when the derivative of the distance is 0.
    # E.g. `d/dt * sqrt((p_x+t*v_x)^2+(p_y+t*v_y)^2+(p_z+t*v_z)^2)=0`
    # This is equivalent to
    #    `d/dt * (p_x+t*v_x)^2+(p_y+t*v_y)^2+(p_z+t*v_z)^2=0`
    # => `2*(p_x+t*v_x)*v_x+2*(p_y+t*v_y)*v_y+2*(p_z+t*v_z)*v_z=0`
    # => `p_x*v_x+p_y*v_y+p_z*v_z+t*(v_x^2+v_y^2+v_z^2)=0`
    # => `t=-(p_x*v_x+p_y*v_y+p_z*v_z)/(v_x^2+v_y^2+v_z^2)`
    denom = np.dot(vel_diff, vel_diff)
    if denom == 0:
        t = 0
    else:
        t = -np.dot(pos_diff, vel_diff) / denom

    if t > max_time:
        t = max_time

    # The minimum distance is then the distance at this time.
    curr_dist = np.linalg.norm(pos_diff)
    min_dist = np.linalg.norm(pos_diff + t * vel_diff)

    return curr_dist, min_dist, t
