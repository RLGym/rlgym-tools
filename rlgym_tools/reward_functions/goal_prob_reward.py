from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM

from rlgym_tools.math.ball import GOAL_THRESHOLD
from rlgym_tools.math.solid_angle import view_goal_ratio


class GoalProbReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.prob = None

    def calculate_blue_goal_prob(self, state: GameState):
        """
        Calculate the probability of a goal being scored *by blue*, e.g. on the orange goal, from the current state.

        :param state: the current game state
        :return: the probability of a goal being scored by blue
        """
        raise NotImplementedError

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prob = self.calculate_blue_goal_prob(initial_state)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                prob = 1
            else:
                prob = 0
        else:
            prob = self.calculate_blue_goal_prob(state)
        # Probability goes from 0-1, but for a reward we want it to go from -1 to 1
        # 2x-1 - (2y-1) = 2(x-y)
        reward = 2 * (prob - self.prob)
        rewards = {
            agent: reward if state.cars[agent].is_blue else -reward
            for agent in agents
        }
        self.prob = prob
        return rewards


class GoalViewReward(GoalProbReward):
    def calculate_blue_goal_prob(self, state: GameState):
        ball_pos = state.ball.position
        view_blue = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)  # Blue net aka orange scoring
        view_orange = view_goal_ratio(ball_pos, GOAL_THRESHOLD)  # Orange net aka blue scoring
        return view_orange / (view_blue + view_orange)
