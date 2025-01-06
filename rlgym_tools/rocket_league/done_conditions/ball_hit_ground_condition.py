from typing import List, Dict, Any

from rlgym.api import DoneCondition, AgentID
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.math.ball import ball_hit_ground


class BallHitGroundCondition(DoneCondition[AgentID, GameState]):
    def __init__(self):
        self.last_tick_count = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_tick_count = initial_state.tick_count

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        ticks_passed = state.tick_count - self.last_tick_count
        done = ball_hit_ground(ticks_passed, state.ball, pre=False)

        self.last_tick_count = state.tick_count

        return {agent: done for agent in agents}
