import time

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.common_values import TICKS_PER_SECOND
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator

from rlgym_tools.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym_tools.action_parsers.delayed_action import DelayedAction
from rlgym_tools.done_conditions.game_condition import GameCondition
from rlgym_tools.math.gamma import half_life_to_gamma
from rlgym_tools.obs_builders.relative_default_obs import RelativeDefaultObs
from rlgym_tools.reward_functions.aerial_distance_reward import AerialDistanceReward
from rlgym_tools.reward_functions.demo_reward import DemoReward
from rlgym_tools.reward_functions.flip_reset_reward import FlipResetReward
from rlgym_tools.reward_functions.goal_prob_reward import GoalViewReward
from rlgym_tools.reward_functions.team_spirit_reward_wrapper import TeamSpiritRewardWrapper
from rlgym_tools.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardProvider
from rlgym_tools.state_mutators.augment_mutator import AugmentMutator
from rlgym_tools.state_mutators.config_mutator import ConfigMutator
from rlgym_tools.state_mutators.game_mutator import GameMutator
from rlgym_tools.state_mutators.replay_mutator import ReplayMutator
from rlgym_tools.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator
from rlgym_tools.state_mutators.weighted_sample_mutator import WeightedSampleMutator


def main():
    tick_skip = 12
    gamma = half_life_to_gamma(half_life_seconds=10, tick_skip=tick_skip)

    env = RLGym(
        state_mutator=MutatorSequence(
            ConfigMutator(boost_consumption=0.1),
            VariableTeamSizeMutator({(1, 1): 2 / 9, (2, 2): 5 / 9, (3, 3): 2 / 9}),
            WeightedSampleMutator.from_zipped(
                # (ReplayMutator(), 0.5),  # TODO: Implement ReplayMutator
                (GameMutator(), 0.5),
            ),
            AugmentMutator()
        ),
        obs_builder=RelativeDefaultObs(),
        action_parser=DelayedAction(
            RepeatAction(
                AdvancedLookupTableAction(),
                repeats=tick_skip),
            action_queue_size=3
        ),
        reward_fn=TeamSpiritRewardWrapper(
            CombinedReward(
                (GoalReward(), 20.),
                (AerialDistanceReward(touch_height_weight=1 / 2044,
                                      car_distance_weight=1 / 5120,
                                      ball_distance_weight=1 / 5120), 1.),
                (DemoReward(), 5.),
                (FlipResetReward(), 10.),
                (GoalViewReward(), 10.),
                (VelocityPlayerToBallReward(), tick_skip / TICKS_PER_SECOND)
            ), team_spirit=0.5),
        transition_engine=RocketSimEngine(),
        termination_cond=GameCondition(seconds_per_goal_forfeit=10, max_overtime_seconds=300),
        truncation_cond=None,
        shared_info_provider=ScoreboardProvider(),
        renderer=None,
    )

    while True:
        obs = env.reset()
        done = False
        t0 = time.perf_counter()
        n = 0
        while not done:
            actions = {
                agent: np.random.randint(0, env.action_space(agent)[1], size=1)
                for agent in obs.keys()
            }
            obs, rewards, is_terminated, is_truncated = env.step(actions)
            n += 1

            assert np.isclose(sum(rewards.values()), 0), "Team spirit reward failed"

            done = any(is_terminated.values()) or any(is_truncated.values())
        t1 = time.perf_counter()
        print(f"Episode done.\n"
              f"\t{n} steps in {t1 - t0:.1f}s ({n / (t1 - t0):.1f} sps)\n"
              f"\t{n * tick_skip} physics ticks ({n * tick_skip / (t1 - t0):.1f} tps)\n"
              f"\t{n * len(actions)} agent steps ({n * len(actions) / (t1 - t0):.1f} sps)\n"
              f"\t{env.shared_info['scoreboard']}")


if __name__ == '__main__':
    main()
