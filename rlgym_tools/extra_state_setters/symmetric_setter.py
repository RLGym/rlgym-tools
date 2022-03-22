from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BALL_RADIUS, CEILING_Z, BLUE_TEAM
import numpy as np
from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_CAR = 1900
PITCH_MAX = np.pi / 2
ROLL_MAX = np.pi


class KickoffLikeSetter(StateSetter):

    def __init__(self, cars_on_ground: bool = True, ball_on_ground: bool = True):
        """
        RandomState constructor.


        :param cars_on_ground: Boolean indicating whether cars should only be placed on the ground.
        :param ball_on_ground: Boolean indicating whether ball should only be placed on the ground.
        """
        super().__init__()
        self.cars_on_ground = cars_on_ground
        self.ball_on_ground = ball_on_ground
        self.yaw_vector = np.asarray([-1, 0, 0])

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_random(state_wrapper, self.ball_on_ground)
        self._reset_cars_random(state_wrapper, self.cars_on_ground)

    def _reset_ball_random(self, state_wrapper: StateWrapper, ball_grounded):
        """
        Function to set the ball to a random position.

        :param state_wrapper: StateWrapper object to be modified.
        :param ball_grounded: Boolean indicating whether ball should only be placed on the ground.
        """
        state_wrapper.ball.set_pos(rand.random() * X_MAX - X_MAX / 2, 0,
                                   BALL_RADIUS if ball_grounded else rand.random() * (
                                       CEILING_Z - 2 * BALL_RADIUS) + BALL_RADIUS)

    def _reset_cars_random(self, state_wrapper: StateWrapper, on_ground: bool):
        """
        Function to set all cars to a random position.

        :param state_wrapper: StateWrapper object to be modified.
        :param on_ground: Boolean indicating whether to place cars only on the ground.
        """
        for i, car in enumerate(state_wrapper.cars):
            # set random position and rotation for all cars based on pre-determined ranges
            if car.team_num == BLUE_TEAM:

                car.set_pos(rand.random() * X_MAX - X_MAX / 2, -abs(rand.random() * Y_MAX - Y_MAX / 2),
                            rand.random() * Z_MAX_CAR + 150)

                # compute vector from ball to car
                rel_ball_car_vector = car.position - state_wrapper.ball.position

                # calculate the angle between the yaw vector and relative vector and use that as yaw.
                yaw = np.arccos(np.dot(rel_ball_car_vector / np.linalg.norm(rel_ball_car_vector), self.yaw_vector))

                # then sprinkle in more variation by offsetting by random angle up to pi/8 radians (this is arbitrary)
                max_offset_angle = np.pi / 8
                yaw += rand.random() * max_offset_angle * 2 - max_offset_angle

                car.set_rot(rand.random() * PITCH_MAX - PITCH_MAX / 2, yaw, rand.random() * ROLL_MAX - ROLL_MAX / 2)

                car.boost = np.random.uniform(0.2, 1)

                # 100% of cars will be set on ground if on_ground == True
                # otherwise, 50% of cars will be set on ground
                if on_ground or rand.random() < 0.5:
                    # z position (up/down) is set to ground
                    car.set_pos(z=17)

                    # pitch (front of car up/down) set to 0
                    # roll (side of car up/down) set to 0
                    car.set_rot(pitch=0, roll=0, yaw=yaw)
                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
            else:
                # the cars in state_wrapper.cars are in order, starting with blue so we can compute blue first and then
                # just copy to orange cars, to assure symmetry
                car_to_copy = state_wrapper.cars[i - len(state_wrapper.cars) // 2]
                car.set_pos(car_to_copy.position[0], -car_to_copy.position[1], car_to_copy.position[2])
                car.set_rot(car_to_copy.rotation[0], -car_to_copy.rotation[1], -car_to_copy.rotation[2])
                car.boost = car_to_copy.boost
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
