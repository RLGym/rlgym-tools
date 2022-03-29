from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

BALL_RADIUS = 94
DEG_TO_RAD = 3.14159265 / 180

class WallPracticeState(StateSetter):

    def __init__(self):
        """
        WallPracticeState to setup wall practice

        """
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        scenario_pick = random.randrange(2)

        if scenario_pick == 0:
            self._short_goal_roll(state_wrapper)
        elif scenario_pick == 1:
            self._side_high_roll(state_wrapper)

    def _side_high_roll(self, state_wrapper):
        """
        A high vertical roll up the side of the field

        :param state_wrapper:
        """
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1


        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = 0
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]
        wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]

        #blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 400 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        #orange car setup
        orange_pitch_rot = 0 * DEG_TO_RAD
        orange_yaw_rot = -90 * DEG_TO_RAD
        orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
        wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

        orange_x = 4096 * side_inverter
        orange_y = 2500 + (random.randrange(500) - 250)
        orange_z = 400 + (random.randrange(400) - 200)
        wall_car_orange.set_pos(orange_x, orange_y, orange_z)
        wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927/180), 0)

    def _short_goal_roll(self, state_wrapper):
        """
        A short roll across the backboard and down in front of the goal

        :param state_wrapper:
        :return:
        """

        defense_team = random.randrange(2)
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 1:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = 1000 * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)


        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]
        challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 25

        challenge_car.set_pos(0, 1000 * defense_inverter, 0)

        challenge_pitch_rot = 0 * DEG_TO_RAD
        challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
        challenge_roll_rot = 0 * DEG_TO_RAD
        challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
        challenge_car.boost = 100

        for car in state_wrapper.cars:
            if car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)
