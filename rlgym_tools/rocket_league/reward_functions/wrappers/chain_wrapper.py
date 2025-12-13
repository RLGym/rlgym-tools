from statistics import fmean
from typing import Callable, Self

from rlgym_tools.rocket_league.reward_functions.wrappers.base_wrapper import BaseWrapper
from rlgym_tools.rocket_league.reward_functions.wrappers.distribute_rewards_wrapper import DistributeRewardsWrapper
from rlgym_tools.rocket_league.reward_functions.wrappers.weighted_wrapper import WeightedWrapper


class ChainWrapper(BaseWrapper):
    """
    A wrapper to easily access the other wrappers by calling their respective methods

    You can call it like:

    ChainWrapper(MyReward()).distribute_rewards().weight(2.0)

    This would give you a distributed THEN weighted reward
    """
    def distribute_rewards(
            self,
            selflessness: float = 1.0,  # aka team_spirit
            selfishness: float = None,  # Defaults to 1 - selflessness
            team_coef: float = 0.5,
            opp_coef: float = None,  # Defaults to 1 - team_coef
            agg_method: Callable[[list[float]], float] = fmean
    ) -> Self:
        """
        Implements a reward distribution scheme.
        It is mainly inspired the distribution scheme described in the OpenAI Five paper (Berner et al., 2019).
        This wrapper has some extra options to allow for more flexibility in reward distribution.

        The adjusted reward for each agent is calculated as follows:
        adjusted_reward = team_coef * (selfishness * own_reward
                                   + selflessness * agg_method(team_rewards))
                          - opp_coef * agg_method(opponent_rewards)
        where:
        - own_reward: The base reward received by the agent.
        - team_aggregate_reward: The mean (or sum) of rewards of all teammates.
        - opponent_aggregate_reward: The mean (or sum) of rewards of all opponents.
        - selflessness: Weight for the team's aggregate reward. Corresponds to the original "team spirit" concept.
        - selfishness: Weight for the agent's own reward (default is 1 - selflessness).
        - team_coef: Coefficient for the team component (default is 0.5).
        - opp_coef: Coefficient for the opponent component (default is 1 - team_coef).
        - agg_method: Method to aggregate team/opponent rewards (default is mean).

        As a default, it distributes rewards equally among teammates and makes it zero-sum with opponents.
        """

        self.reward_function = DistributeRewardsWrapper(
            self.reward_function,
            selflessness=selflessness,
            selfishness=selfishness,
            team_coef=team_coef,
            opp_coef=opp_coef,
            agg_method=agg_method
        )
        return self

    def weight(self, weight: float = 1.0) -> Self:
        """
        Weights a reward based on the following formula:

        reward * weight
        """
        self.reward_function = WeightedWrapper(
            reward_function=self.reward_function,
            weight=weight
        )
        return self