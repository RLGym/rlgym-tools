from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED

import numpy as np


class DemoReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, attacker_reward=1, victim_punishment=1):
        self.attacker_reward = attacker_reward
        self.victim_punishment = victim_punishment

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        return

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        for agent in agents:
            victim = state.cars[agent].bump_victim_id
            if victim is not None:
                if state.cars[victim].demo_respawn_timer > 0:
                    rewards[agent] += self.attacker_reward
                    rewards[victim] -= self.victim_punishment

        return rewards
