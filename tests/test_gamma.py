import math

from rlgym.rocket_league.common_values import TICKS_PER_SECOND

from rlgym_tools.math.gamma import half_life_to_gamma, horizon_to_gamma, gamma_to_half_life, gamma_to_horizon


def main():
    for tick_skip in (4, 6, 8, 12):
        for gamma in (0.9, 0.99, 0.999):
            assert math.isclose(
                half_life_to_gamma(gamma_to_half_life(gamma, tick_skip), tick_skip),
                gamma
            )
            assert math.isclose(
                horizon_to_gamma(gamma_to_horizon(gamma, tick_skip), tick_skip),
                gamma
            )
            assert math.isclose(
                gamma ** (gamma_to_half_life(gamma, tick_skip) * TICKS_PER_SECOND / tick_skip),
                0.5
            )
            assert math.isclose(
                gamma_to_horizon(gamma, tick_skip) * TICKS_PER_SECOND / tick_skip,
                1 / (1 - gamma),
            )
        for duration in (5, 10, 15, 30):
            assert math.isclose(
                gamma_to_half_life(half_life_to_gamma(duration, tick_skip), tick_skip),
                duration
            )
            assert math.isclose(
                gamma_to_horizon(horizon_to_gamma(duration, tick_skip), tick_skip),
                duration
            )


if __name__ == '__main__':
    main()
