import math

import numpy as np

from rlgym.rocket_league.api import PhysicsObject
from rlgym.rocket_league.common_values import TICKS_PER_SECOND, GRAVITY, BACK_WALL_Y, BALL_RADIUS, GOAL_CENTER_TO_POST, \
    BALL_RESTING_HEIGHT, GOAL_THRESHOLD


# Utility functions for ball physics


def solve_parabolic_trajectory(ball: PhysicsObject, g=GRAVITY):
    # Solve for the time of impact with the ground
    z = ball.position[2] - BALL_RESTING_HEIGHT
    vz = ball.linear_velocity[2]
    a = -0.5 * g
    b = vz
    c = z
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None, None
    discriminant = math.sqrt(discriminant)
    t_neg = (-b + discriminant) / (2 * a)
    t_pos = (-b - discriminant) / (2 * a)
    return t_neg, t_pos


def ball_hit_ground(ticks_passed: int, ball: PhysicsObject, pre=False):
    if ball.position[2] <= BALL_RESTING_HEIGHT:
        return True
    t_neg, t_pos = solve_parabolic_trajectory(ball)
    if t_neg is None or t_pos is None:
        return False
    if pre:
        # Positive solution, e.g. looking in the future
        # Less accurate since ball can hit something else before the ground.
        if 0 <= t_pos <= ticks_passed / TICKS_PER_SECOND:
            return True
    else:
        # Negative solution, e.g. looking in the past.
        # Generally preferred as it's more likely to be correct.
        if -ticks_passed / TICKS_PER_SECOND <= t_neg <= 0:
            return True
    return False


def closest_point_in_goal(ball_pos, margin=BALL_RADIUS):
    # Find the closest point on each goal to the ball
    x = math.copysign(1, ball_pos[0]) * min(abs(ball_pos[0]), GOAL_CENTER_TO_POST - margin)
    y = BACK_WALL_Y + margin
    z = min(ball_pos[2], GOAL_THRESHOLD)
    return np.array([x, y, z])
