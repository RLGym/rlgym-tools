from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM

from rlgym_tools.math.ball import GOAL_THRESHOLD
from rlgym_tools.math.solid_angle import view_goal_ratio


class GoalProbReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.prob = None

    def calculate_goal_prob(self, state: GameState):
        raise NotImplementedError

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prob = self.calculate_goal_prob(initial_state)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                prob = 1
            else:
                prob = 0
        else:
            prob = self.calculate_goal_prob(state)
        # Probability goes from 0-1, but for a reward we want it to go from -1 to 1
        # 2x-1 - (2y-1) = 2(x-y)
        reward = 2 * (prob - self.prob)
        rewards = {
            agent: reward if state.cars[agent].is_blue else -reward
            for agent in agents
        }
        return rewards


class GoalViewReward(GoalProbReward):
    def calculate_goal_prob(self, state: GameState):
        ball_pos = state.ball.position
        view_blue = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)
        view_orange = view_goal_ratio(ball_pos, GOAL_THRESHOLD)
        odds_blue = view_blue / (1 - view_blue)
        odds_orange = view_orange / (1 - view_orange)
        return 1 / (1 + odds_blue / odds_orange)
