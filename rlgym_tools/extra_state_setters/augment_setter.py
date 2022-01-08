import math
from copy import deepcopy
from random import getrandbits, shuffle
from typing import List

import numpy as np
from rlgym.utils.state_setters.state_setter import StateSetter
from rlgym.utils.state_setters.wrappers import CarWrapper
from rlgym.utils.state_setters.wrappers import StateWrapper

PI = math.pi


class AugmentSetter(StateSetter):
    MASK_SWAP_FRONT_BACK = 0b01
    MASK_SWAP_LEFT_RIGHT = 0b10

    def __init__(self, state_setter: StateSetter, shuffle_within_teams=True, swap_front_back=True,
                 swap_left_right=True):
        super().__init__()
        self.state_setter = state_setter
        self.shuffle_within_teams = shuffle_within_teams
        self.swap_front_back = swap_front_back
        self.swap_left_right = swap_left_right

    def reset(self, state_wrapper: StateWrapper):
        self.state_setter.reset(state_wrapper)
        # self._debug(state_wrapper)

        bits = getrandbits(2)
        if self.shuffle_within_teams:
            self.shuffle_players(state_wrapper)

        if self.swap_front_back and (bits & AugmentSetter.MASK_SWAP_FRONT_BACK):
            self.mirror_front_back(state_wrapper)

        if self.swap_left_right and (bits & AugmentSetter.MASK_SWAP_LEFT_RIGHT):
            self.mirror_left_right(state_wrapper)

        # self._debug(state_wrapper)

    def _debug(self, state_wrapper: StateWrapper):
        print("\n".join(f"Car {car.id}, team: {car.team_num}, pos: {car.position}" for car in state_wrapper.cars))
        ball = state_wrapper.ball
        print(f"Ball pos: {ball.position}")

    @staticmethod
    def _map_cars(cars0: List[CarWrapper], cars1: List[CarWrapper]):
        # Transfer all the states from cars0 to cars1
        for car0, car1 in zip(cars0, cars1):
            if cars0 == car1:  # By reference
                continue
            car0.position[:], car1.position[:] = \
                car1.position[:].copy(), car0.position[:].copy()
            car0.linear_velocity[:], car1.linear_velocity[:] = \
                car1.linear_velocity[:].copy(), car0.linear_velocity[:].copy()
            car0.rotation[:], car1.rotation[:] = \
                car1.rotation[:].copy(), car0.rotation[:].copy()
            car0.angular_velocity[:], car1.angular_velocity[:] = \
                car1.angular_velocity[:].copy(), car0.angular_velocity[:].copy()
            car0.boost, car1.boost = \
                car1.boost, car0.boost

    @staticmethod
    def shuffle_players(state_wrapper: StateWrapper):
        """ The cars within a team are randomly swapped with each other """
        if len(state_wrapper.cars) <= 2:
            return
        blue_team = deepcopy(state_wrapper.blue_cars())
        orange_team = deepcopy(state_wrapper.orange_cars())
        shuffle(blue_team)
        shuffle(orange_team)

        AugmentSetter._map_cars(state_wrapper.blue_cars(), blue_team)
        AugmentSetter._map_cars(state_wrapper.orange_cars(), orange_team)

    @staticmethod
    def switch_teams(state_wrapper):
        """ Blue cars move to Orange positions, orange to blue """
        AugmentSetter._map_cars(state_wrapper.orange_cars(), state_wrapper.blue_cars())

    @staticmethod
    def mirror_front_back(state_wrapper: StateWrapper):
        AugmentSetter.switch_teams(state_wrapper)
        mul = np.array([1, -1, 1])

        for obj in [state_wrapper.ball] + state_wrapper.cars:
            obj.set_pos(*(mul * obj.position))
            obj.set_lin_vel(*(mul * obj.linear_velocity))
            obj.set_ang_vel(*(-mul * obj.angular_velocity))  # Angular velocities are negated

            if isinstance(obj, CarWrapper):
                pitch, yaw, roll = obj.rotation
                obj.set_rot(
                    pitch=pitch,
                    yaw=-yaw,
                    roll=-roll,
                )

    @staticmethod
    def mirror_left_right(state_wrapper: StateWrapper):
        mul = np.array([-1, 1, 1])

        for obj in [state_wrapper.ball] + state_wrapper.cars:
            obj.set_pos(*(mul * obj.position))
            obj.set_lin_vel(*(mul * obj.linear_velocity))
            obj.set_ang_vel(*(-mul * obj.angular_velocity))  # Angular velocities are negated

            if isinstance(obj, CarWrapper):
                pitch, yaw, roll = obj.rotation
                obj.set_rot(
                    pitch=pitch,
                    yaw=PI - yaw,
                    roll=-roll,
                )
