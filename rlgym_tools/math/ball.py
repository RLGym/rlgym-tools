import math

import numpy as np
from rlgym.rocket_league.api import PhysicsObject
from rlgym.rocket_league.common_values import TICKS_PER_SECOND, GRAVITY, BACK_WALL_Y, BALL_RADIUS, GOAL_HEIGHT, \
    GOAL_CENTER_TO_POST

BALL_RESTING_HEIGHT = 93.15
GOAL_THRESHOLD = 5215.5  # Tested in-game with BakkesMod


def ball_hit_ground(ticks_passed: int, ball: PhysicsObject):
    z = ball.position[2] - BALL_RESTING_HEIGHT
    if z < 0:
        return True

    vz = ball.linear_velocity[2]

    # Reverse the trajectory to find the time of impact
    g = GRAVITY
    a = -0.5 * g
    b = vz
    c = z
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return False
    t = (-b - discriminant ** 0.5) / (2 * a)  # Negative solution since we're looking for the past
    if -ticks_passed / TICKS_PER_SECOND <= t <= 0:
        return True
    return False


def closest_point_in_goal(ball_pos, margin=BALL_RADIUS):
    # Find the closest point on each goal to the ball
    x = math.copysign(1, ball_pos[0]) * min(abs(ball_pos[0]), GOAL_CENTER_TO_POST - margin)
    y = BACK_WALL_Y + margin
    z = min(ball_pos[2], GOAL_THRESHOLD)
    return np.array([x, y, z])
