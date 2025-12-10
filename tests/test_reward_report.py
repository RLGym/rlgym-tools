import copy
from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import GoalReward

from rlgym_tools.rocket_league.math.gamma import horizon_to_gamma
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.reward_report import get_reward_df, generate_report, generate_plot
from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward
from rlgym_tools.rocket_league.reward_functions.ball_travel_reward import BallTravelReward
from rlgym_tools.rocket_league.reward_functions.boost_change_reward import BoostChangeReward
from rlgym_tools.rocket_league.reward_functions.boost_keep_reward import BoostKeepReward
from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
from rlgym_tools.rocket_league.reward_functions.flip_reset_reward import FlipResetReward
from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import GoalViewReward
from rlgym_tools.rocket_league.reward_functions.team_spirit_reward_wrapper import TeamSpiritRewardWrapper
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward, \
    TrajectoryComparisonVPBReward
from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward


class EpisodeEnd(RewardFunction):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: is_terminated[agent] or is_truncated[agent]
                for agent in agents}


def main():
    parsed_replay = ParsedReplay.load("test_replays/c3e4895a-b536-4a39-9e7b-5fa88ffb9a07.replay")
    rewards_list = [
        (EpisodeEnd(), ""),
        (FlipResetReward(1, 0), "/obtain"),
        (FlipResetReward(0, 1), "/hit"),
        (DemoReward(), ""),
        (DemoReward(0, 0, 1), "/bump_only"),
        (WavedashReward(), ""),
        (AdvancedTouchReward(), ""),
        (AdvancedTouchReward(0, 1), "/acc_only"),
        # (BallTravelReward(), ""),
        (BallTravelReward(1, 0.5, 0.5, -1, 0., do_integral=True), "/zero_sum_ready"),
        (BallTravelReward(1, 0, 0, 0, 0, 0, 0, do_integral=True), "/integral"),
        (BallTravelReward(1, 0, 0, 0, 0, 0, 0), "/consecutive"),
        (BallTravelReward(0, 1, 0, 0, 0, 0, 0), "/pass"),
        (BallTravelReward(0, 0, 1, 0, 0, 0, 0), "/receive"),
        (BallTravelReward(0, 0, 0, 1, 0, 0, 0), "/giveaway"),
        (BallTravelReward(0, 0, 0, 0, 1, 0, 0), "/intercept"),
        (BallTravelReward(0, 0, 0, 0, 0, 1, 0), "/goal"),
        (BallTravelReward(1, 0.5, 0.5, 0, 0, 1, 0), "/possession"),
        (GoalViewReward(), ""),
        (AerialDistanceReward(1, 0, 0), "/touch_height"),
        (AerialDistanceReward(0, 1, 0), "/car_distance"),
        (AerialDistanceReward(0, 0, 1), "/ball_distance"),
        (VelocityPlayerToBallReward(), ""),
        (VelocityPlayerToBallReward(False), "/no_negative"),
        (TrajectoryComparisonVPBReward(True), "/trajectory_comparison"),
        (BoostChangeReward(), ""),
        (BoostKeepReward(), ""),
        (BoostKeepReward(activation_fn=lambda x: x), "/identity"),
        (BoostKeepReward(activation_fn=lambda x: (1 - 0.1 ** x) / 0.9), "/exp"),
        (GoalReward(), "")
    ]
    # Add zero-sum versions
    rewards_list += [
        (TeamSpiritRewardWrapper(copy.deepcopy(r), 0.0), t + "_zero_sum")
        for r, t in rewards_list
    ]

    meta_df, reward_df = get_reward_df(
        parsed_replay,
        rewards_list
    )

    generate_report(meta_df, reward_df)

    fig = generate_plot(meta_df, reward_df, gamma=horizon_to_gamma(horizon_seconds=15, tick_skip=4))
    fig.savefig("reward_report.png")

    debug = True


if __name__ == '__main__':
    main()
