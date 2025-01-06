from typing import Any, Dict, List

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class TeamSpiritRewardWrapper(RewardFunction[AgentID, GameState, float]):
    """
    Implements team spirit, as described in the OpenAI Five paper (Berner et al., 2019).
    """
    def __init__(self, reward_fn: RewardFunction, team_spirit: float):
        self.reward_fn = reward_fn
        self.team_spirit = team_spirit

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        base_rewards = self.reward_fn.get_rewards(list(state.cars.keys()), state, is_terminated, is_truncated, shared_info)

        # We calculate the means first
        total_blue = total_orange = n_blue = n_orange = 0
        for agent, car in state.cars.items():
            if car.is_blue:
                total_blue += base_rewards[agent]
                n_blue += 1
            else:
                total_orange += base_rewards[agent]
                n_orange += 1
        mean_blue = total_blue / n_blue if n_blue > 0 else 0
        mean_orange = total_orange / n_orange if n_orange > 0 else 0

        # Then do the team spirit adjustment
        spirit_rewards = {}
        for agent in agents:
            rew = base_rewards[agent]
            if state.cars[agent].is_blue:
                team_mean, opp_mean = mean_blue, mean_orange
            else:
                team_mean, opp_mean = mean_orange, mean_blue
            # Actual formula here
            spirit_rewards[agent] = (1 - self.team_spirit) * rew + self.team_spirit * team_mean - opp_mean

        return spirit_rewards
