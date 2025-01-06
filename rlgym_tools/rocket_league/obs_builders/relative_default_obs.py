import math
from typing import List, Dict, Any, Tuple

import numpy as np

from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import ORANGE_TEAM
from rlgym_tools.rocket_league.math.relative import relative_physics, dodge_relative_physics


class RelativeDefaultObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """
    Relative version of the default observation builder.
    Agent's car uses absolute reference frame, other cars and ball are relative to the agent's car.
    """

    def __init__(self, zero_padding=3, pos_coef=1 / 2300, ang_coef=1 / math.pi, lin_vel_coef=1 / 2300,
                 ang_vel_coef=1 / math.pi,
                 pad_timer_coef=1 / 10, boost_coef=1 / 100, dodge_relative=False):
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
        self.BOOST_COEF = boost_coef
        self.zero_padding = zero_padding
        self.relative_fn = relative_physics if not dodge_relative else dodge_relative_physics

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
        if car.team_num == ORANGE_TEAM:
            physics = car.inverted_physics
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            physics = car.physics
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        rel_ball, *rel_cars = relative_physics(car.physics,
                                               [state.ball] + [car.physics for car in state.cars.values()])

        obs = [  # Global stuff
            rel_ball.position * self.POS_COEF,
            rel_ball.linear_velocity * self.LIN_VEL_COEF,
            rel_ball.angular_velocity * self.ANG_VEL_COEF,
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

        car_obs = self._make_player_obs(physics, car)
        obs.append(car_obs)

        allies = []
        enemies = []

        for (other, other_car), other_physics in zip(state.cars.items(), rel_cars):
            if other == agent:
                continue

            if other_car.team_num == car.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            team_obs.append(self._make_player_obs(other_physics, other_car))

        if self.zero_padding is not None:
            # Padding for multi game mode
            while len(allies) < self.zero_padding - 1:
                allies.append(np.zeros_like(car_obs))
            while len(enemies) < self.zero_padding:
                enemies.append(np.zeros_like(car_obs))

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _make_player_obs(self, physics: PhysicsObject, car: Car) -> np.ndarray:
        return np.concatenate([
            physics.position * self.POS_COEF,
            physics.forward,
            physics.up,
            physics.linear_velocity * self.LIN_VEL_COEF,
            physics.angular_velocity * self.ANG_VEL_COEF,
            [car.boost_amount * self.BOOST_COEF,
             car.demo_respawn_timer,
             int(car.on_ground),
             int(car.is_boosting),
             int(car.is_supersonic)]
        ])
