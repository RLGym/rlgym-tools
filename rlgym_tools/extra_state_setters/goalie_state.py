from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi


class GoaliePracticeState(StateSetter):

    def __init__(self, aerial_only=False, allow_enemy_interference=False, first_defender_in_goal=False,
                 reset_to_max_boost=True):
        """
        GoaliePracticeState constructor.

        :param aerial_only: Boolean indicating whether the shots will only be in the air.
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0  # swap every reset who's getting shot at

        self.aerial_only = aerial_only
        self.allow_enemy_interference = allow_enemy_interference
        self.first_defender_in_goal = first_defender_in_goal
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new shot

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball(state_wrapper, self.team_turn, self.aerial_only)
        self._reset_cars(state_wrapper, self.team_turn, self.first_defender_in_goal, self.allow_enemy_interference,
                         self.reset_to_max_boost)

        # which team will recieve the next incoming shot
        self.team_turn = (self.team_turn + 1) % 2

    def _reset_cars(self, state_wrapper: StateWrapper, team_turn, first_defender_in_goal, allow_enemy_interference,
                    reset_to_max_boost):
        """
        Function to set cars in preparation for an incoming shot

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set:
                if first_defender_in_goal:
                    y_pos = -GOAL_LINE if car.team_num == 0 else GOAL_LINE
                    car.set_pos(0, y_pos, z=17)
                    first_set = True
                else:
                    self._place_car_in_box_area(car, team_turn)

            else:
                if allow_enemy_interference:
                    self._place_car_in_box_area(car, team_turn)

                else:
                    self._place_car_in_box_area(car, car.team_num)

            if reset_to_max_boost:
                car.boost = 100

            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX / 2, 0)

    def _place_car_in_box_area(self, car, team_delin):
        """
        Function to place a car in an allowed areaI 

        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        y_pos = (PLACEMENT_BOX_Y - (rand.random() * PLACEMENT_BOX_Y))

        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET

        car.set_pos(rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X / 2, y_pos, z=17)

    def _reset_ball(self, state_wrapper: StateWrapper, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """

        pos, lin_vel, ang_vel = self._get_shot_parameters(team_turn, aerial_only)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

    def _get_shot_parameters(self, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal
        
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """

        # *** Magic numbers are from manually calibrated shots ***
        # *** They are unrelated to numbers in other functions ***

        shotpick = random.randrange(4)
        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1)  # invert shot for blue

        # random pick x value of target in goal
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)

        # if its not an air shot, we can randomize the shot speed
        shot_randomizer = 1 if aerial_only else (random.uniform(.6, 1))

        y_vel = (3000 * INVERT_IF_BLUE) if aerial_only else (3000 * shot_randomizer * INVERT_IF_BLUE)
        if shotpick == 0:  # long range shot

            z_pos = 1500 if aerial_only else random.uniform(100, 1500)

            pos = np.array([x_pos, -3300 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 600])
        elif shotpick == 1:  # medium range shot
            z_pos = 1550 if aerial_only else random.uniform(100, 1550)

            pos = np.array([x_pos, -500 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 100])

        elif shotpick == 2:  # angled shot
            z_pos = 1500 if aerial_only else random.uniform(100, 1500)
            x_pos += 3200  # add offset to start the shot from the side
            y_pos = -2000 * INVERT_IF_BLUE

            x_vel = -1100
            y_vel = 2500 * INVERT_IF_BLUE

            pos = np.array([x_pos, y_pos, z_pos])
            lin_vel = np.array([x_vel, y_vel, 650])

        elif shotpick == 3:  # opposite angled shot
            z_pos = 1500 if aerial_only else random.uniform(100, 1500)
            x_pos -= 3200  # add offset to start the shot from the other side
            y_pos = 2000 * INVERT_IF_BLUE

            x_vel = 1100
            y_vel = -2500 * INVERT_IF_BLUE

            pos = np.array([x_pos, y_pos, z_pos])
            lin_vel = np.array([x_vel, y_vel, 650])
        else:
            print("FAULT")

        ang_vel = np.array([0, 0, 0])

        return pos, lin_vel, ang_vel
