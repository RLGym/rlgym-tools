import math
from typing import List, Dict, Any, Callable

from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class BoostKeepReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward_per_second: float = 1.0,
                 activation_fn: Callable[[float], float] = lambda x: math.sqrt(0.01 * x)):
        """
        Reward function that rewards agents for having boost in their tank.

        :param reward_per_second: Amount of reward to give per second at full boost.
        :param activation_fn: Activation function to apply to the boost value before calculating the reward. Default is
                              the square root function so that increasing boost is more important when boost is low.
        """
        self.reward_per_tick = reward_per_second / TICKS_PER_SECOND
        self.activation_fn = activation_fn

        self.prev_ticks = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ticks = initial_state.tick_count

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        ticks_passed = state.tick_count - self.prev_ticks
        mul = self.reward_per_tick * ticks_passed
        rewards = {}
        for agent in agents:
            boost = state.cars[agent].boost_amount
            rewards[agent] = self.activation_fn(boost) * mul
        self.prev_ticks = state.tick_count

        return rewards
