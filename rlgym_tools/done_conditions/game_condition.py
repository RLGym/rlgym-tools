from typing import List, Dict, Any

from rlgym.api import DoneCondition, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, BALL_RADIUS
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class GameCondition(DoneCondition[AgentID, GameState]):
    def __init__(self, game_duration_seconds: int, seconds_per_goal_forfeit=None, max_overtime_seconds=None):
        self.game_duration_seconds = game_duration_seconds
        self.seconds_left = game_duration_seconds
        self.is_overtime = False
        self.seconds_per_goal_forfeit = seconds_per_goal_forfeit
        self.max_overtime_seconds = max_overtime_seconds
        self.scoreline = (0, 0)
        self.prev_state = None

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.seconds_left = self.game_duration_seconds
        self.is_overtime = False
        self.scoreline = (0, 0)
        self.prev_state = initial_state
        shared_info["scoreboard"] = {"scoreline": self.scoreline,
                                     "is_overtime": self.is_overtime,
                                     "seconds_left": self.seconds_left,
                                     "go_to_kickoff": True}

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        ticks_passed = state.tick_count - self.prev_state.tick_count
        self.seconds_left -= ticks_passed / TICKS_PER_SECOND
        self.seconds_left = max(0, self.seconds_left)
        dones = {agent: False for agent in agents}
        go_to_kickoff = False
        if state.goal_scored:
            if self.is_overtime or self.seconds_left <= 0:
                dones = {agent: True for agent in agents}
            if state.scoring_team == BLUE_TEAM:
                self.scoreline = (self.scoreline[0] + 1, self.scoreline[1])
            else:
                self.scoreline = (self.scoreline[0], self.scoreline[1] + 1)
        elif self.seconds_left <= 0:
            prev_ball = self.prev_state.ball
            next_z = prev_ball.position[2] + ticks_passed * prev_ball.velocity[2] / TICKS_PER_SECOND
            if next_z - BALL_RADIUS < 0:  # Ball would be below the ground
                if self.scoreline[0] != self.scoreline[1]:
                    dones = {agent: True for agent in agents}
                else:
                    go_to_kickoff = True
                    self.is_overtime = True
        elif self.seconds_per_goal_forfeit is not None:
            goal_diff = abs(self.scoreline[0] - self.scoreline[1])
            if goal_diff >= 3:
                seconds_per_goal = self.seconds_left / goal_diff
                if seconds_per_goal < self.seconds_per_goal_forfeit:  # Forfeit if it's not realistic to catch up
                    dones = {agent: True for agent in agents}

        self.prev_state = state
        shared_info["scoreboard"] = {"scoreline": self.scoreline,
                                     "is_overtime": self.is_overtime,
                                     "seconds_left": self.seconds_left,
                                     "go_to_kickoff": go_to_kickoff}
        return dones
