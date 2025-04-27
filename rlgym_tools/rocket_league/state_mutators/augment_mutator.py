import random
from copy import deepcopy
from typing import Dict, Any

import numpy as np

from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, BOOST_LOCATIONS

boost_locations = np.array(BOOST_LOCATIONS)
_left_right_pad_indices = np.empty(len(boost_locations), dtype=int)
_front_back_pad_indices = np.empty(len(boost_locations), dtype=int)
for i in range(len(boost_locations)):
    _left_right_pad_indices[i] = np.argmin(
        np.linalg.norm(boost_locations - boost_locations[i] * [-1, 1, 1], axis=1))
    _front_back_pad_indices[i] = np.argmin(
        np.linalg.norm(boost_locations - boost_locations[i] * [1, -1, 1], axis=1))


class AugmentMutator(StateMutator[GameState]):
    def __init__(self, shuffle_within_teams: bool = True,
                 randomize_front_back: bool = True, randomize_left_right: bool = True):
        self.shuffle_within_teams = shuffle_within_teams
        self.randomize_front_back = randomize_front_back
        self.randomize_left_right = randomize_left_right

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        bits = random.getrandbits(2)

        scoreboard = shared_info.get("scoreboard")

        if self.shuffle_within_teams:
            self.shuffle_players(state)
        if self.randomize_front_back and bits & 1:
            self.swap_front_back(state)
            if scoreboard is not None:
                scoreboard.blue_score, scoreboard.orange_score = scoreboard.orange_score, scoreboard.blue_score
        if self.randomize_left_right and bits & 2:
            self.swap_left_right(state)

    @staticmethod
    def shuffle_players(state):
        for team in BLUE_TEAM, ORANGE_TEAM:
            agents = [agent for agent, car in state.cars.items() if car.team_num == team]
            if len(agents) == 1:
                continue
            from_agents = list(agents)
            to_agents = list(agents)
            random.shuffle(to_agents)
            mapping = dict(zip(from_agents, to_agents))
            original_cars = {k: v for k, v in state.cars.items()}
            for from_agent, to_agent in mapping.items():
                from_car = original_cars.pop(from_agent)
                if from_car.bump_victim_id is not None:
                    from_car.bump_victim_id = mapping[from_car.bump_victim_id]
                state.cars[to_agent] = from_car

    @staticmethod
    def swap_front_back(state):  # Across x-axis, meaning y-axis is inverted
        state.ball = deepcopy(state.ball)  # Circumvents a bug in KickoffMutator in base library
        state.ball.position[1] *= -1
        state.ball.linear_velocity[1] *= -1
        state.ball.angular_velocity[0] *= -1
        state.ball.angular_velocity[2] *= -1
        for car in state.cars.values():
            car.team_num = BLUE_TEAM if car.is_orange else ORANGE_TEAM
            car.physics = deepcopy(car.physics)  # Circumvents a bug in KickoffMutator in base library
            car.physics.position[1] *= -1
            car.physics.linear_velocity[1] *= -1
            car.physics.angular_velocity[0] *= -1
            car.physics.angular_velocity[2] *= -1
            rot_mtx = car.physics.rotation_mtx.copy()
            # Invert y-axis for fwd and up vectors, and x+z axes for right vector (to maintain handedness)
            rot_mtx[1, 0] *= -1
            rot_mtx[0, 1] *= -1
            rot_mtx[2, 1] *= -1
            rot_mtx[1, 2] *= -1
            car.physics.rotation_mtx = rot_mtx
            wwc = car.wheels_with_contact
            car.wheels_with_contact = wwc[1], wwc[0], wwc[3], wwc[0]  # Swap wheels
            car.flip_torque[0] *= -1  # Roll torque
            car.autoflip_direction *= -1
        state.boost_pad_timers[:] = state.boost_pad_timers[_front_back_pad_indices]

    @staticmethod
    def swap_left_right(state):  # Across y-axis, meaning x-axis is inverted
        state.ball = deepcopy(state.ball)  # Circumvents a bug in KickoffMutator in base library
        state.ball.position[0] *= -1
        state.ball.linear_velocity[0] *= -1
        state.ball.angular_velocity[1] *= -1
        state.ball.angular_velocity[2] *= -1
        for car in state.cars.values():
            car.physics = deepcopy(car.physics)  # Circumvents a bug in KickoffMutator in base library
            car.physics.position[0] *= -1
            car.physics.linear_velocity[0] *= -1
            car.physics.angular_velocity[1] *= -1
            car.physics.angular_velocity[2] *= -1
            rot_mtx = car.physics.rotation_mtx.copy()
            # Invert x-axis for fwd and up vectors, and y+z axes for right vector (to maintain handedness)
            rot_mtx[0, 0] *= -1
            rot_mtx[1, 1] *= -1
            rot_mtx[2, 1] *= -1
            rot_mtx[0, 2] *= -1
            car.physics.rotation_mtx = rot_mtx
            wwc = car.wheels_with_contact
            car.wheels_with_contact = wwc[1], wwc[0], wwc[3], wwc[0]  # Swap wheels
            car.flip_torque[0] *= -1  # Roll torque
            car.autoflip_direction *= -1
        state.boost_pad_timers[:] = state.boost_pad_timers[_left_right_pad_indices]
