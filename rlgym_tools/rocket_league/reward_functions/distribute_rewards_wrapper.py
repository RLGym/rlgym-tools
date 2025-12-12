from statistics import fmean
from typing import Any, Dict, List, Callable

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class DistributeRewardsWrapper(RewardFunction[AgentID, GameState, float]):
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

    def __init__(
            self,
            reward_fn: RewardFunction,
            *,
            selflessness: float = 1.0,  # aka team_spirit
            selfishness: float = None,  # Defaults to 1 - selflessness
            team_coef: float = 0.5,
            opp_coef: float = None,  # Defaults to 1 - team_coef
            agg_method: Callable[[list[float]], float] = fmean,
    ):
        self.reward_fn = reward_fn
        self.selflessness = selflessness
        self.selfishness = selfishness if selfishness is not None else (1 - selflessness)
        self.team_coef = team_coef
        self.opp_coef = opp_coef if opp_coef is not None else (1 - team_coef)
        self.agg_method = agg_method

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        base_rewards = self.reward_fn.get_rewards(
            list(state.cars.keys()), state, is_terminated, is_truncated, shared_info
        )

        # We calculate the team aggregates first
        blue_rewards = []
        orange_rewards = []
        for agent, car in state.cars.items():
            if car.is_blue:
                blue_rewards.append(base_rewards[agent])
            else:
                orange_rewards.append(base_rewards[agent])
        agg_blue = self.agg_method(blue_rewards)
        agg_orange = self.agg_method(orange_rewards)

        # Then we distribute the rewards
        adjusted_rewards = {}
        for agent in agents:
            rew = base_rewards[agent]
            if state.cars[agent].is_blue:
                team_agg, opp_agg = agg_blue, agg_orange
            else:
                team_agg, opp_agg = agg_orange, agg_blue
            # Actual formula here
            spirit_reward = self.selfishness * rew + self.selflessness * team_agg
            adjusted_reward = self.team_coef * spirit_reward - self.opp_coef * opp_agg
            adjusted_rewards[agent] = adjusted_reward

        return adjusted_rewards
