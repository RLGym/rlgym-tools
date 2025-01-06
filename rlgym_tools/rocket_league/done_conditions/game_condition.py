from typing import List, Dict, Any

from rlgym.api import DoneCondition, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, BALL_RADIUS
from rlgym.rocket_league.common_values import TICKS_PER_SECOND

from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo


class GameCondition(DoneCondition[AgentID, GameState]):
    """
    Simulates a Rocket League game, ending when:
    - A goal is scored
    - The ball hits the ground at 0 seconds
    - The overtime exceeds the maximum allowed time
    - The scoreline is presumed to be insurmountable, leading to a forfeit
    """

    def __init__(self, seconds_per_goal_forfeit=None, max_overtime_seconds=None):
        self.seconds_per_goal_forfeit = seconds_per_goal_forfeit
        self.max_overtime_seconds = max_overtime_seconds
        self.overtime_duration = 0
        self.last_ticks = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.overtime_duration = 0
        self.last_ticks = initial_state.tick_count

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        scoreboard: ScoreboardInfo = shared_info["scoreboard"]

        done = False
        if scoreboard.go_to_kickoff or scoreboard.is_over:
            done = True
        elif scoreboard.is_overtime:
            self.overtime_duration += (state.tick_count - self.last_ticks) / TICKS_PER_SECOND
            if self.max_overtime_seconds is not None and self.overtime_duration >= self.max_overtime_seconds:
                scoreboard.is_over = True
                done = True
        else:
            if self.seconds_per_goal_forfeit is not None:
                goal_diff = abs(scoreboard.blue_score - scoreboard.orange_score)
                if goal_diff >= 3:
                    seconds_per_goal = scoreboard.game_timer_seconds / goal_diff
                    if seconds_per_goal < self.seconds_per_goal_forfeit:  # Forfeit if it's not realistic to catch up
                        scoreboard.is_over = True
                        done = True

        self.last_ticks = state.tick_count

        return {agent: done for agent in agents}
