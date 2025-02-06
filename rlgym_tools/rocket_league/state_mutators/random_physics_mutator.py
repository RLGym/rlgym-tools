from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from rlgym.rocket_league.math import rand_vec3, rand_uvec3

from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import RAMP_HEIGHT


class RandomPhysicsMutator(StateMutator[GameState]):
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 100  # Ball radius and car hitbox with biggest radius are both below this
        for po in [state.ball] + [car.physics for car in state.cars.values()]:
            while True:
                new_pos = np.random.uniform([-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, 0 + padding],
                                            [SIDE_WALL_X - padding, BACK_WALL_Y - padding, CEILING_Z - padding])

                # Some checks to make sure we don't place it outside the field
                if abs(new_pos[0]) + abs(new_pos) >= 8064 - padding:
                    continue
                close_to_wall = (abs(new_pos[0]) >= SIDE_WALL_X - RAMP_HEIGHT
                                 or abs(new_pos[1]) >= BACK_WALL_Y - RAMP_HEIGHT
                                 or abs(new_pos[0]) + abs(new_pos[1]) >= 8064 - RAMP_HEIGHT)
                close_to_floor_or_ceiling = (new_pos[2] <= RAMP_HEIGHT
                                             or new_pos[2] >= CEILING_Z - RAMP_HEIGHT)
                if close_to_wall and close_to_floor_or_ceiling:
                    continue
                break
            po.position = new_pos
            po.linear_velocity = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)
            fw = rand_uvec3()
            up = rand_uvec3()
            right = np.cross(fw, up)
            po.rotation_mtx = np.stack([fw, right, up])
