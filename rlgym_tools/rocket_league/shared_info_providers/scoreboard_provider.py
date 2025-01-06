import math
from dataclasses import dataclass, asdict, replace
from typing import List, Dict, Any

from rlgym.api import SharedInfoProvider, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import TICKS_PER_SECOND, BLUE_TEAM

from rlgym_tools.rocket_league.math.ball import ball_hit_ground


@dataclass(slots=True)
class ScoreboardInfo:
    game_timer_seconds: float
    kickoff_timer_seconds: float
    blue_score: int
    orange_score: int
    go_to_kickoff: bool
    is_over: bool

    @property
    def is_overtime(self):
        return math.isinf(self.game_timer_seconds)

    @property
    def is_kickoff(self):
        return self.kickoff_timer_seconds > 0


class ScoreboardProvider(SharedInfoProvider[AgentID, GameState]):
    """
    Implements a Rocket League scoreboard, since RLGym does not provide one.
    It tracks:
    - The current scoreline
    - The time remaining
    - The time remaining until kickoff buffer ends
    - Whether the game should go to kickoff
    - Whether the game is over
    Your state setter is responsible for initializing the scoreboard with the desired values.
    """

    def __init__(self):
        self.last_ticks = None
        self.info = None

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        if "scoreboard" not in shared_info:
            # Default behavior is infinite overtime, basically the same as no scoreboard
            shared_info["scoreboard"] = ScoreboardInfo(
                game_timer_seconds=math.inf,
                kickoff_timer_seconds=0.,
                blue_score=0,
                orange_score=0,
                go_to_kickoff=False,
                is_over=True,
            )
        return shared_info

    def set_state(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        info = shared_info.get("scoreboard")
        if info is None:
            raise ValueError("ScoreboardProvider requires a 'scoreboard' key in shared_info")

        if isinstance(info, dict):
            info = ScoreboardInfo(**info)

        assert not any(v is None for v in asdict(info).values()), "ScoreboardInfo must be fully defined"

        if info.is_over:
            raise ValueError("Cannot set scoreboard to be over")

        self.info = info

        self.last_ticks = initial_state.tick_count

        shared_info["scoreboard"] = info
        return shared_info

    def step(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        ticks_passed = state.tick_count - self.last_ticks
        self.last_ticks = state.tick_count

        # Copy info into new object to avoid modifying in place
        info: ScoreboardInfo = replace(self.info)

        if info.go_to_kickoff or info.is_over:
            ticks_passed = 0

        info.go_to_kickoff = False
        info.is_over = False

        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                info.blue_score += 1
            else:
                info.orange_score += 1
            if info.is_overtime:
                info.is_over = True
            info.go_to_kickoff = True

        if info.kickoff_timer_seconds > 0 and state.ball.position[1] == 0:
            info.kickoff_timer_seconds -= ticks_passed / TICKS_PER_SECOND
            if info.kickoff_timer_seconds <= 0:
                info.kickoff_timer_seconds = 0
        else:
            info.kickoff_timer_seconds = 0
            info.game_timer_seconds -= ticks_passed / TICKS_PER_SECOND

        if info.game_timer_seconds < 0:
            info.game_timer_seconds = 0
            if ball_hit_ground(ticks_passed, state.ball) or state.goal_scored:
                if info.blue_score == info.orange_score:
                    info.game_timer_seconds = math.inf
                    info.go_to_kickoff = True
                else:
                    info.is_over = True

        self.info = info
        shared_info["scoreboard"] = self.info
        return shared_info
