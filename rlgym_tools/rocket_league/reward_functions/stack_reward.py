from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType
from rlgym.rocket_league.api import GameState


class StackReward(RewardFunction[AgentID, GameState, np.ndarray]):
    """
    Turns the results from several reward functions into a single vector with each reward.
    Might be useful if someone wants to compute values and do some custom post-processing on them,
    e.g. adaptive weighting, normalization, logging.
    """
    def __init__(self, reward_functions: List[RewardFunction[AgentID, GameState, float]]):
        self.reward_functions = reward_functions

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        for reward_function in self.reward_functions:
            reward_function.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        rewards = {agent: [] for agent in agents}
        for reward_function in self.reward_functions:
            for agent, reward in reward_function.get_rewards(agents, state, is_terminated, is_truncated,
                                                             shared_info).items():
                rewards[agent].append(reward)
        return {agent: np.stack(rewards[agent]) for agent in agents}
