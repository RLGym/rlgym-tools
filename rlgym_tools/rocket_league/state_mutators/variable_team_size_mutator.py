from typing import Dict, Any

import numpy as np
import random

from rlgym.api import StateMutator
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BLUE_TEAM, OCTANE, ORANGE_TEAM


class VariableTeamSizeMutator(StateMutator[GameState]):

    def __init__(self, mode_weights: Dict[tuple, float]):
        self.mode_weights = mode_weights

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        assert len(state.cars) == 0, "VariableTeamSizeMutator should be applied before any cars are added to the state."
        keys, values = zip(*self.mode_weights.items())
        mode = random.choices(keys, values)[0]
        blue_size, orange_size = mode

        for idx in range(blue_size):
            car = self.make_new_car()
            car.team_num = BLUE_TEAM
            state.cars['blue-{}'.format(idx)] = car

        for idx in range(orange_size):
            car = self.make_new_car()
            car.team_num = ORANGE_TEAM
            state.cars['orange-{}'.format(idx)] = car

    def make_new_car(self) -> Car:
        """
        Users can override this to set default values.
        """
        car = Car()
        car.hitbox_type = OCTANE

        car.physics = PhysicsObject()

        car.demo_respawn_timer = 0.
        car.on_ground = True
        car.supersonic_time = 0.
        car.boost_amount = 0.
        car.boost_active_time = 0.
        car.handbrake = 0.

        car.has_jumped = False
        car.is_holding_jump = False
        car.is_jumping = False
        car.jump_time = 0.

        car.has_flipped = False
        car.has_double_jumped = False
        car.air_time_since_jump = 0.
        car.flip_time = 0.
        car.flip_torque = np.zeros(3, dtype=np.float32)

        car.is_autoflipping = False
        car.autoflip_timer = 0.
        car.autoflip_direction = 0.

        return car
