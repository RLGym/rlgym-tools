from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM

from rlgym_tools.rocket_league.math.ball import GOAL_THRESHOLD
from rlgym_tools.rocket_league.math.solid_angle import view_goal_ratio


class GoalProbReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, gamma: float = 1):
        """
        According to Ng. et al. (1999), a reward shaping function must be of the form:
        F(s, a, s') = γ * Φ(s') - Φ(s)
        to preserve all the optimal policies of the original MDP,
        where Φ(s) is a function that estimates the potential of a state.
        The gamma term is supposed to be the same as the one used to discount future rewards.
        Here it serves to adjust for the fact that it will be discounted in the future.
        In practice though, leaving it as 1 is probably fine.
        (in fact the paper only deals with finite MDPs with γ=1 and infinite MDPs with γ<1,
        whereas we typically have a finite MDP with γ<1)

        :param gamma: the discount factor for the reward shaping function.
        """
        self.prob = None
        self.gamma = gamma

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
        reward = 2 * (self.gamma * prob - self.prob)
        rewards = {
            agent: reward if state.cars[agent].is_blue else -reward
            for agent in agents
        }
        self.prob = prob
        return rewards


class GoalViewReward(GoalProbReward):
    """
    Simple estimate based on the apparent size of each goal.
    Basically it says "if we cast a ray from the ball in random directions until it hits a goal,
    what's the chance it hits the orange goal (blue scoring)?"
    """

    def calculate_blue_goal_prob(self, state: GameState):
        ball_pos = state.ball.position
        view_blue = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)  # Blue net aka orange scoring
        view_orange = view_goal_ratio(ball_pos, GOAL_THRESHOLD)  # Orange net aka blue scoring
        return view_orange / (view_blue + view_orange)
