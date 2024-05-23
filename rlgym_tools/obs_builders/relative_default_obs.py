import math
from typing import List, Dict, Any, Tuple

import numpy as np

from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM


class RelativeDefaultObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """
    Relative version of the default observation builder.
    Agent's car uses absolute reference frame, other cars are relative to the agent's car.
    """

    def __init__(self, zero_padding=3, pos_coef=1 / 2300, ang_coef=1 / math.pi, lin_vel_coef=1 / 2300,
                 ang_vel_coef=1 / math.pi,
                 pad_timer_coef=1 / 10):
        """
        :param zero_padding: Number of max cars per team, if not None the obs will be zero padded
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        :param pad_timer_coef: Boost pad timers normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.zero_padding = zero_padding

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        if self.zero_padding is not None:
            return 'real', 52 + 20 * self.zero_padding * 2
        else:
            return 'real', -1  # Without zero padding this depends on the initial state, but we don't want to crash for now

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[
        AgentID, np.ndarray]:
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state, shared_info)

        return obs

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        car = state.cars[agent]
        if car.is_orange:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        obs = [  # Global stuff
            ball.position * self.POS_COEF,
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads * self.PAD_TIMER_COEF,
            [  # Partially observable variables
                car.is_holding_jump,
                car.handbrake,
                car.has_jumped,
                car.is_jumping,
                car.has_flipped,
                car.is_flipping,
                car.has_double_jumped,
                car.can_flip,
                car.air_time_since_jump
            ]
        ]

        car_obs = self._generate_absolute_car_obs(car, inverted)
        obs.append(car_obs)

        allies = []
        enemies = []

        for other, other_car in state.cars.items():
            if other == agent:
                continue

            if other_car.team_num == car.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            team_obs.append(self._generate_relative_car_obs(car, other_car))

        if self.zero_padding is not None:
            # Padding for multi game mode
            while len(allies) < self.zero_padding - 1:
                allies.append(np.zeros_like(car_obs))
            while len(enemies) < self.zero_padding:
                enemies.append(np.zeros_like(car_obs))

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _generate_absolute_car_obs(self, car: Car, inverted: bool) -> np.ndarray:
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        return np.concatenate([
            physics.position * self.POS_COEF,
            physics.forward,
            physics.up,
            physics.linear_velocity * self.LIN_VEL_COEF,
            physics.angular_velocity * self.ANG_VEL_COEF,
            [car.boost_amount,
             car.demo_respawn_timer,
             int(car.on_ground),
             int(car.is_boosting),
             int(car.is_supersonic)]
        ])

    def _generate_relative_car_obs(self, base_car: Car, other_car: Car) -> np.ndarray:
        # No inverted since team inversion does not affect relative positions
        rot_mtx = base_car.physics.rotation_mtx

        relative_pos = (other_car.physics.position - base_car.physics.position) @ rot_mtx
        relative_forward = other_car.physics.forward @ rot_mtx
        relative_up = other_car.physics.up @ rot_mtx
        relative_lin_vel = (other_car.physics.linear_velocity - base_car.physics.linear_velocity) @ rot_mtx
        relative_ang_vel = (other_car.physics.angular_velocity - base_car.physics.angular_velocity) @ rot_mtx

        return np.concatenate([
            relative_pos * self.POS_COEF,
            relative_forward,
            relative_up,
            relative_lin_vel * self.LIN_VEL_COEF,
            relative_ang_vel * self.ANG_VEL_COEF,
            [other_car.boost_amount,
             other_car.demo_respawn_timer,
             int(other_car.on_ground),
             int(other_car.is_boosting),
             int(other_car.is_supersonic)]
        ])
