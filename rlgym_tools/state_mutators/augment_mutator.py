import random
from typing import Dict, Any

from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM


class AugmentMutator(StateMutator[GameState]):
    def __init__(self, shuffle_within_teams: bool = True, swap_front_back: bool = True, swap_left_right: bool = True):
        self.shuffle_within_teams = shuffle_within_teams
        self.swap_front_back = swap_front_back
        self.swap_left_right = swap_left_right

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        bits = random.getrandbits(2)

        if self.shuffle_within_teams:
            self.shuffle_players(state)
        if self.swap_front_back and bits & 1:
            self.mirror_across_x(state)
        if self.swap_left_right and bits & 2:
            self.mirror_across_y(state)

    @staticmethod
    def shuffle_players(state):
        for team in BLUE_TEAM, ORANGE_TEAM:
            agents = [agent for agent, car in state.cars.items() if car.team == team]
            from_agent = list(agents)
            to_agent = list(agents)
            random.shuffle(to_agent)
            mapping = dict(zip(from_agent, to_agent))
            for from_agent, to_agent in mapping.items():
                from_car = state.cars.pop(from_agent)
                if from_car.bump_victim_id is not None:
                    from_car.bump_victim_id = mapping[from_car.bump_victim_id]
                state.cars[to_agent] = from_car

    @staticmethod
    def mirror_across_x(state):  # Across x-axis, meaning y-axis is inverted
        for car in state.cars.values():
            car.team_num = BLUE_TEAM if car.team_num == ORANGE_TEAM else ORANGE_TEAM
            car.physics.position[1] *= -1
            car.physics.velocity[1] *= -1
            car.physics.angular_velocity[0] *= -1
            car.physics.angular_velocity[2] *= -1
            rot_mtx = car.physics.rotation_mtx.copy()
            # Invert y-axis for fwd and up vectors, and x+z axes for right vector (to maintain handedness)
            rot_mtx[1, 0] *= -1
            rot_mtx[0, 1] *= -1
            rot_mtx[2, 1] *= -1
            rot_mtx[1, 2] *= -1
            car.physics.rotation_mtx = rot_mtx

    @staticmethod
    def mirror_across_y(state):  # Across y-axis, meaning x-axis is inverted
        for car in state.cars.values():
            car.physics.position[0] *= -1
            car.physics.velocity[0] *= -1
            car.physics.angular_velocity[1] *= -1
            car.physics.angular_velocity[2] *= -1
            rot_mtx = car.physics.rotation_mtx.copy()
            # Invert x-axis for fwd and up vectors, and y+z axes for right vector (to maintain handedness)
            rot_mtx[0, 0] *= -1
            rot_mtx[1, 1] *= -1
            rot_mtx[2, 1] *= -1
            rot_mtx[0, 2] *= -1
            car.physics.rotation_mtx = rot_mtx
