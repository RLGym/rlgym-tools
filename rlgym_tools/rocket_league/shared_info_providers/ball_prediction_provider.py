from typing import List, Dict, Any

import RocketSim as rs
import numpy as np
from rlgym.api import SharedInfoProvider, AgentID
from rlgym.rocket_league.api import GameState, PhysicsObject
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class BallPredictionProvider(SharedInfoProvider[AgentID, GameState]):
    def __init__(self, limit_seconds, step_seconds, game_mode=rs.GameMode.SOCCAR):
        self.limit_ticks = int(limit_seconds * TICKS_PER_SECOND)
        self.step_ticks = int(step_seconds * TICKS_PER_SECOND)
        assert self.limit_ticks % self.step_ticks == 0, "limit_seconds must be divisible by step_seconds"
        self.ball_predictor = rs.BallPredictor(game_mode)
        self.ball_prediction = None
        self.prev_tick_count = 0

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        if "ball_prediction" not in shared_info:
            shared_info["ball_prediction"] = None
        return shared_info

    def _update_prediction(self, ball: PhysicsObject, ticks_passed: int):
        rs_ball_prediction = self.ball_predictor.get_ball_prediction(
            rs.BallState(
                pos=rs.Vec(*ball.position),
                rot_mat=rs.RotMat(*ball.rotation_mtx.transpose().flatten()),
                vel=rs.Vec(*ball.linear_velocity),
                ang_vel=rs.Vec(*ball.angular_velocity)
            ),
            ticks_passed,
            self.limit_ticks // self.step_ticks,
            self.step_ticks
        )
        ball_prediction = [PhysicsObject() for _ in range(len(rs_ball_prediction))]
        for ball, prediction in zip(ball_prediction, rs_ball_prediction):
            ball.position = np.array([prediction.pos.x, prediction.pos.y, prediction.pos.z])
            ball.rotation_mtx = prediction.rot_mat.as_numpy().T
            ball.linear_velocity = np.array([prediction.vel.x, prediction.vel.y, prediction.vel.z])
            ball.angular_velocity = np.array([prediction.ang_vel.x, prediction.ang_vel.y, prediction.ang_vel.z])
            ball_prediction.append(ball)
        self.ball_prediction = ball_prediction

    def set_state(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        self._update_prediction(initial_state.ball, 0)
        shared_info["ball_prediction"] = self.ball_prediction
        return shared_info

    def step(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        self._update_prediction(state.ball, state.tick_count - self.prev_tick_count)
        self.prev_tick_count = state.tick_count
        shared_info["ball_prediction"] = self.ball_prediction
        return shared_info
