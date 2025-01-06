import math

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, SIDE_WALL_X, GOAL_HEIGHT

try:
    from scipy.stats.distributions import skellam
except ImportError:
    skellam = None  # Lets us import the constants without scipy. It'll fail if you try to use the function.

# Win probability estimation using Skellam distribution

AVERAGE_EPISODE_LENGTH = np.array([26.32, 47.16, 68.52])  # In seconds
GOALS_PER_MINUTE = 60 / AVERAGE_EPISODE_LENGTH

FLOOR_AREA = 4 * BACK_WALL_Y * SIDE_WALL_X - 1152 * 1152  # Subtract corners
GOAL_AREA = GOAL_HEIGHT * 880
BASE_FLOOR_PROB = FLOOR_AREA / (2 * GOAL_AREA + FLOOR_AREA)
SECONDS_PER_MINUTE = 60


def win_prob(goals_per_minute: float, time_left_seconds: float, differential: int,
             next_goal_prob: float = 0.5, hit_ground_prob: float = BASE_FLOOR_PROB):
    if skellam is None:
        raise ImportError("scipy is required to use the win_prob function")
    # First, make it symmetric, so we only need to concern ourselves with non-negative differentials
    if differential < 0:
        p_loss = win_prob(goals_per_minute, time_left_seconds, -differential,
                          1 - next_goal_prob, hit_ground_prob)
        return 1 - p_loss

    if math.isinf(time_left_seconds):
        # Overtime, next goal wins
        return next_goal_prob
    if differential == 0 and next_goal_prob == 0.5:
        return 0.5

    # We win if we score, or go to OT and win there (50%)
    win_prob_if_tied = (1 - hit_ground_prob) * next_goal_prob + 0.5 * hit_ground_prob
    # We win if ball hits the ground or we score
    win_prob_if_one_ahead = hit_ground_prob + (1 - hit_ground_prob) * next_goal_prob
    # We win only by scoring without it hitting the ground
    win_prob_if_one_behind = (1 - hit_ground_prob) * next_goal_prob

    p_two_goal_lead = p_one_goal_lead = p_tied = p_one_goal_deficit = 0
    if time_left_seconds <= 0:
        if differential == 0:
            p_tied = 1
        elif differential == 1:
            p_one_goal_lead = 1
        else:  # differential >= 2:
            p_two_goal_lead = 1
    else:
        rate = goals_per_minute * time_left_seconds / SECONDS_PER_MINUTE

        dist = skellam(rate / 2, rate / 2)
        p_two_goal_lead = dist.cdf(differential - 2)
        p_one_goal_lead, p_tied, p_one_goal_deficit = dist.pmf([differential - 1, differential, differential + 1])

    p_win = (
            p_two_goal_lead  # No way for opponent to win
            + p_one_goal_lead * win_prob_if_one_ahead
            + p_tied * win_prob_if_tied
            + p_one_goal_deficit * win_prob_if_one_behind
    )

    return p_win
