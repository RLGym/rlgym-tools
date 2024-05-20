from dataclasses import dataclass
from typing import List, Dict, Any

from rlgym.api import SharedInfoProvider, AgentID
from rlgym.rocket_league.api import GameState, PhysicsObject
from rlgym.rocket_league.common_values import TICKS_PER_SECOND, GRAVITY, BLUE_TEAM

PRE_MATCH = 0
KICKOFF = 1
REGULATION = 2
OVERTIME = 3
POST_MATCH = 4

BALL_RESTING_HEIGHT = 93.15


@dataclass
class ScoreboardInfo:
    seconds_remaining: float
    blue_score: int
    orange_score: int
    game_state: int


class ScoreboardProvider(SharedInfoProvider[AgentID, GameState]):
    """
    Implements a Rocket League scoreboard, since RLGym does not provide one.
    It tracks:
    - The current scoreline
    - The time remaining
    - The current game state (e.g. overtime, kickoff)
    """

    def __init__(self, game_length_seconds: float = 300.0):
        self.game_length_seconds = game_length_seconds
        self.ticks_remaining = game_length_seconds * TICKS_PER_SECOND
        self.blue_score = 0
        self.orange_score = 0
        self.game_state = PRE_MATCH

    def get_info(self):
        info = ScoreboardInfo(self.ticks_remaining * TICKS_PER_SECOND,
                              self.blue_score, self.orange_score,
                              self.game_state)
        return info

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        self.game_state = PRE_MATCH
        self.blue_score = 0
        self.orange_score = 0
        self.ticks_remaining = self.game_length_seconds * TICKS_PER_SECOND
        shared_info["scoreboard"] = self.get_info()
        return shared_info

    def set_state(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        if initial_state.ball.position[1] == 0:
            self.game_state = KICKOFF
        else:
            self.game_state = REGULATION

    @staticmethod
    def ball_hit_ground(ticks_passed: int, ball: PhysicsObject):
        z = ball.position[2] - BALL_RESTING_HEIGHT
        if z < 0:
            return True

        vz = ball.linear_velocity[2]

        # Reverse the trajectory to find the time of impact
        g = GRAVITY
        a = -0.5 * g
        b = vz
        c = z
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False
        t = (-b - discriminant ** 0.5) / (2 * a)  # Negative solution since we're looking for the past
        if -ticks_passed / TICKS_PER_SECOND <= t <= 0:
            return True
        return False

    def step(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        ticks_passed = state.tick_count

        if self.game_state == REGULATION:
            self.ticks_remaining -= ticks_passed
            self.ticks_remaining = max(0, self.ticks_remaining)

        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                self.blue_score += 1
            else:
                self.orange_score += 1

        if self.ticks_remaining <= 0:
            if self.ball_hit_ground(ticks_passed, state.ball) or state.goal_scored:
                if self.blue_score == self.orange_score:
                    self.game_state = OVERTIME
                    self.ticks_remaining = float("inf")
                else:
                    self.game_state = POST_MATCH

        shared_info["scoreboard"] = self.get_info()
        return shared_info
