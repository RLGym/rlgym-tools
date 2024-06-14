from typing import List, Dict, Any

from rlgym.api import SharedInfoProvider, AgentID
from rlgym.rocket_league.api import GameState, PhysicsObject

import RocketSim as rs


class BallPredictionProvider(SharedInfoProvider[AgentID, GameState]):
    def __init__(self, num_ticks, tick_skip, arena=None):
        self.num_ticks = num_ticks
        self.tick_skip = tick_skip
        if arena is None:
            arena = rs.Arena(rs.GameMode.SOCCAR)
        self.arena = arena
        self.ball_prediction = None

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        if "ball_prediction" not in shared_info:
            shared_info["ball_prediction"] = None
        return shared_info

    def _set_ball_state(self, ball: PhysicsObject):
        self.arena.ball.set_state(
            rs.BallState(
                pos=rs.Vec(*ball.position),
                vel=rs.Vec(*ball.linear_velocity),
                rot_mat=rs.RotMat(*ball.rotation_mtx.transpose().flatten())
            )
        )

    def _update_prediction(self):
        self.ball_prediction = self.arena.get_ball_prediction(self.num_ticks, self.tick_skip)

    def set_state(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        self._set_ball_state(initial_state.ball)
        self._update_prediction()
        shared_info["ball_prediction"] = self.ball_prediction
        return shared_info

    def step(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        self._set_ball_state(state.ball)
        self._update_prediction()
        shared_info["ball_prediction"] = self.ball_prediction
        return shared_info
