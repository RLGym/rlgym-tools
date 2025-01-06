import math

from rlgym.rocket_league.common_values import TICKS_PER_SECOND


# Utility functions for converting between gamma, half-life, and horizon.

def half_life_to_gamma(half_life_seconds: float, tick_skip: int) -> float:
    half_life_ticks = half_life_seconds * TICKS_PER_SECOND
    half_life_steps = half_life_ticks / tick_skip
    gamma = 0.5 ** (1 / half_life_steps)
    # gamma^half_life_steps = 0.5
    return gamma


def gamma_to_half_life(gamma: float, tick_skip: int) -> float:
    half_life_steps = -math.log(2) / math.log(gamma)
    half_life_ticks = half_life_steps * tick_skip
    half_life_seconds = half_life_ticks / TICKS_PER_SECOND
    return half_life_seconds


def horizon_to_gamma(horizon_seconds: float, tick_skip: int) -> float:
    horizon_ticks = horizon_seconds * TICKS_PER_SECOND
    horizon_steps = horizon_ticks / tick_skip
    gamma = 1 - 1 / horizon_steps
    # 1/(1-gamma) = horizon_steps
    return gamma


def gamma_to_horizon(gamma: float, tick_skip: int) -> float:
    horizon_steps = 1 / (1 - gamma)
    horizon_ticks = horizon_steps * tick_skip
    horizon_seconds = horizon_ticks / TICKS_PER_SECOND
    return horizon_seconds
