import math

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, BALL_RADIUS, GOAL_HEIGHT, GOAL_CENTER_TO_POST


def closest_point_in_goal(ball_pos):
    # Find the closest point on each goal to the ball
    x = math.copysign(1, ball_pos[0]) * min(abs(ball_pos[0]), GOAL_CENTER_TO_POST - BALL_RADIUS)
    y = BACK_WALL_Y + BALL_RADIUS
    z = min(ball_pos[2], GOAL_HEIGHT - BALL_RADIUS)
    return np.array([x, y, z])


def solid_angle_eriksson(O, A, B, C):
    # Calculate the solid angle of a triangle ABC from the point O
    a = A - O
    b = B - O
    c = C - O
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c /= np.linalg.norm(c)
    numerator = np.linalg.norm(np.dot(np.cross(a, b), c))  # noqa numpy complains about cross for no reason
    denominator = 1 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    E = 2 * math.atan2(numerator, denominator)
    return E


def view_goal_ratio(pos, goal_y):
    # Calculate the percent of the field of view that the goal takes up
    max_x = GOAL_CENTER_TO_POST - BALL_RADIUS
    max_y = GOAL_HEIGHT - BALL_RADIUS
    bl = np.array([-max_x, goal_y, BALL_RADIUS])
    br = np.array([max_x, goal_y, BALL_RADIUS])
    tl = np.array([-max_x, goal_y, max_y])
    tr = np.array([max_x, goal_y, max_y])
    solid_angle_1 = solid_angle_eriksson(pos, bl, br, tl)
    solid_angle_2 = solid_angle_eriksson(pos, br, tr, tl)
    return (solid_angle_1 + solid_angle_2) / (4 * math.pi)


def solid_angle_ball(pos, ball_pos, ball_radius=BALL_RADIUS):
    # Calculate the solid angle of the ball
    d = np.linalg.norm(pos - ball_pos)
    r_sphere = math.sqrt(d ** 2 - ball_radius ** 2)
    E = 2 * math.pi * (1 - r_sphere / d)
    return E


def view_ball_ratio(pos, ball_pos):
    # Calculate the percent of the field of view that the ball takes up
    solid_angle = solid_angle_ball(pos, ball_pos)
    return solid_angle / (4 * math.pi)
