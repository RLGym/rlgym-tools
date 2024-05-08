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
        return car
