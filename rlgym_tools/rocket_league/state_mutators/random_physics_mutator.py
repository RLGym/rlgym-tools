from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import rand_vec3, rand_uvec3


class RandomPhysicsMutator(StateMutator[GameState]):
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 100
        for po in [state.ball] + [car.physics for car in state.cars.values()]:
            po.position = np.random.uniform([-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, 0 + padding],
                                            [SIDE_WALL_X - padding, BACK_WALL_Y - padding, CEILING_Z - padding])
            po.linear_velocity = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)
            fw = rand_uvec3()
            up = rand_uvec3()
            right = np.cross(fw, up)
            po.rotation_mtx = np.stack([fw, right, up])
