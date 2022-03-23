from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BALL_RADIUS, CEILING_Z, BLUE_TEAM, ORANGE_TEAM
import numpy as np
from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_CAR = 1900
GOAL_HEIGHT = 642.775
PITCH_MAX = np.pi / 2
ROLL_MAX = np.pi
BLUE_GOAL_POSITION = np.asarray([0, -5120, 0])
ORANGE_GOAL_POSITION = np.asarray([0, 5120, 0])


class HoopsLikeSetter(StateSetter):

    def __init__(self, spawn_radius: float = 800):
        """
        Hoops-like kickoff constructor

        :param spawn_radius: Float determining how far away to spawn the cars from the ball.
        """
        super().__init__()
        self.spawn_radius = spawn_radius
        self.yaw_vector = np.asarray([-1, 0, 0])

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_random(state_wrapper)
        self._reset_cars_random(state_wrapper, self.spawn_radius)

    def _reset_ball_random(self, state_wrapper: StateWrapper):
        """
        Resets the ball according to a uniform distribution along y-axis and normal distribution along x-axis.

        :param state_wrapper: StateWrapper object to be modified.
        """
        new_x = rand.uniform(-4000, 4000)
        new_y = rand.triangular(-5000, 0, 5000)
        new_z = rand.uniform(GOAL_HEIGHT, CEILING_Z - 2 * BALL_RADIUS)
        state_wrapper.ball.set_pos(new_x, new_y, new_z)
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        state_wrapper.ball.set_lin_vel(0, 0, 0)

    def _reset_cars_random(self, state_wrapper: StateWrapper, spawn_radius: float):
        """
        Function to set all cars inbetween the ball and net roughly facing the ball. The other cars will be spawned
        randomly.

        :param state_wrapper: StateWrapper object to be modified.
        :param spawn_radius: Float determining how far away to spawn the cars from the ball.
        """
        orange_done = False
        blue_done = False
        for i, car in enumerate(state_wrapper.cars):
            if car.team_num == BLUE_TEAM and not blue_done:
                # just shorthands for ball_x and ball_y
                bx = state_wrapper.ball.position[0]
                by = state_wrapper.ball.position[1]

                # add small variation to spawn radius
                car_spawn_radius = rand.triangular(0.8, 1, 1.2) * spawn_radius

                # calculate distance from ball to goal.
                R = ((BLUE_GOAL_POSITION[0] - bx) ** 2 + (BLUE_GOAL_POSITION[1] - by) ** 2) ** 0.5

                # use similarity of triangles to calculate offsets
                x_offset = ((BLUE_GOAL_POSITION[0] - bx) / R) * car_spawn_radius
                y_offset = ((BLUE_GOAL_POSITION[1] - by) / R) * car_spawn_radius

                # offset the car's positions
                car.set_pos(bx + x_offset, by + y_offset, 17)

                # compute vector from ball to car
                rel_ball_car_vector = car.position - state_wrapper.ball.position

                # calculate the angle between the yaw vector and relative vector and use that as yaw.

                yaw = np.arccos(np.dot(rel_ball_car_vector / np.linalg.norm(rel_ball_car_vector), self.yaw_vector))

                # then sprinkle in more variation by offsetting by random angle up to pi/8 radians (this is arbitrary)
                max_offset_angle = np.pi / 8
                yaw += rand.random() * max_offset_angle * 2 - max_offset_angle

                # random at least slightly above 0 boost amounts
                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=yaw)

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
                blue_done = True
            elif car.team_num == ORANGE_TEAM and not orange_done:
                # just shorthands for ball_x and ball_y
                bx = state_wrapper.ball.position[0]
                by = state_wrapper.ball.position[1]

                # add small variation to spawn radius
                car_spawn_radius = rand.triangular(0.8, 1, 1.2) * spawn_radius

                # calculate distance from ball to goal.
                R = ((ORANGE_GOAL_POSITION[0] - bx) ** 2 + (ORANGE_GOAL_POSITION[1] - by) ** 2) ** 0.5

                # use similarity of triangles to calculate offsets
                x_offset = ((ORANGE_GOAL_POSITION[0] - bx) / R) * car_spawn_radius
                y_offset = ((ORANGE_GOAL_POSITION[1] - by) / R) * car_spawn_radius

                # offset the car's positions
                car.set_pos(bx + x_offset, by + y_offset, 17)

                # compute vector from ball to car
                rel_ball_car_vector = car.position - state_wrapper.ball.position

                # calculate the angle between the yaw vector and relative vector and use that as yaw.

                yaw = np.arccos(np.dot(rel_ball_car_vector / np.linalg.norm(rel_ball_car_vector), self.yaw_vector))

                # then sprinkle in more variation by offsetting by random angle up to pi/8 radians (this is arbitrary)
                max_offset_angle = np.pi / 8
                yaw += rand.random() * max_offset_angle * 2 - max_offset_angle

                # random at least slightly above 0 boost amounts
                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=yaw - np.pi if yaw > 0 else yaw + np.pi)

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
                orange_done = True
            else:
                car.set_pos(rand.uniform(-4096,4096)),rand.uniform(0,-1**(car.team_num-1)*5120,17)

                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=rand.uniform(-np.pi, np.pi))

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
