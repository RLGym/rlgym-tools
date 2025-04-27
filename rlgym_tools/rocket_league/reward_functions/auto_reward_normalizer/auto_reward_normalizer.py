from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID, StateType
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.reward_functions.auto_reward_normalizer.running_normalizer import RunningNormalizer
from rlgym_tools.rocket_league.reward_functions.auto_reward_normalizer.simple_z_normalizer import SimpleZNormalizer


class AutoRewardNormalizer(RewardFunction[AgentID, GameState, float]):

    # Update modes:
    ON_RESET = 0
    BEFORE_NORM = 2
    AFTER_NORM = 3
    BEFORE_NORM_THEN_ON_RESET = 4

    def __init__(self, reward_function: RewardFunction[AgentID, GameState, float],
                 dist_tracker: RunningNormalizer = None,
                 update_mode: int=None):
        self.reward_function = reward_function
        self.dist_tracker = dist_tracker or SimpleZNormalizer()
        if update_mode is None:
            if isinstance(self.dist_tracker, SimpleZNormalizer):
                self.update_mode = self.BEFORE_NORM_THEN_ON_RESET
            else:
                self.update_mode = self.ON_RESET
        self.update_mode = update_mode

        self._rewards_since_reset = []

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.reward_function.reset(agents, initial_state, shared_info)
        if len(self._rewards_since_reset) > 0:
            if self.update_mode == self.ON_RESET:
                self.dist_tracker.update(self._rewards_since_reset)
            elif self.update_mode == self.BEFORE_NORM_THEN_ON_RESET:
                self.update_mode = self.ON_RESET
        self._rewards_since_reset = []

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = self.reward_function.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        rew_list = [rewards[a] for a in agents]

        match self.update_mode:
            case self.ON_RESET:
                self._rewards_since_reset += rew_list
                norm_rewards = self.dist_tracker.normalize(rew_list)
            case self.BEFORE_NORM:
                self.dist_tracker.update(rew_list)
                norm_rewards = self.dist_tracker.normalize(rew_list)
            case self.AFTER_NORM:
                norm_rewards = self.dist_tracker.normalize(rew_list)
                self.dist_tracker.update(rew_list)
            case self.BEFORE_NORM_THEN_ON_RESET:
                self._rewards_since_reset += rew_list
                self.dist_tracker.update(rew_list)
                norm_rewards = self.dist_tracker.normalize(rew_list)
            case _:
                raise ValueError(f"Unknown update mode: {self.update_mode}")

        norm_rewards = {
            a: norm_rewards[i]
            for i, a in enumerate(agents)
        }

        if any(abs(r) == 10 for r in norm_rewards.values()):
            debug = True

        return norm_rewards
