import random
from typing import List, Union

import numpy as np
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper


def _expand_cars(state_wrapper: StateWrapper, data: np.ndarray):
    rng = np.random.default_rng()
    num_cars_in_replay = (len(data) - 9) // 13
    num_cars_to_add = len(state_wrapper.cars) - num_cars_in_replay
    car_data = np.split(data[9:], num_cars_in_replay)
    item_positions = [data[:3]]
    for i, car in enumerate(car_data):
        item_positions.append(car_data[i][:3])
    # longest dimension of any hitbox is 131.49 (breakout) so check 150 distance for safety
    for i in range(num_cars_to_add):
        checks = 0
        closest_dist = 0
        x = 0
        y = 0
        z = 17
        while closest_dist <= 150:
            x = rng.uniform(-2800, 2800)
            y = rng.uniform(-3800, 3800)
            item_positions.sort(key=lambda p: np.linalg.norm(p - np.array([x, y, z])))
            closest_dist = np.linalg.norm(item_positions[0] - np.array([x, y, z]))
            checks += 1
            # having trouble finding space on the floor, put it in the air
            # this seems extremely rare. I ran 100_000 tests with 6 initial cars and ball at 17
            # and the most checks ever was 4
            if checks > 5:
                z = rng.uniform(140, 1800)
        item_positions.append(np.array([x, y, z]))
        data = np.append(data, [x, y, z, 0, rng.uniform(-np.pi, np.pi), 0, rng.uniform(0, 500), rng.uniform(0, 500),
                                0, 0, 0, 0, rng.uniform(0, 1)])

    return data


def _shrink_cars(state_wrapper: StateWrapper, data: np.ndarray):
    num_cars_in_replay = (len(data) - 9) // 13
    num_cars_to_remove = num_cars_in_replay - len(state_wrapper.cars)
    per_team_to_remove = num_cars_to_remove // 2
    leftover = num_cars_to_remove - (per_team_to_remove * 2)
    blue_end = (num_cars_in_replay // 2) - 1
    orange_end = (num_cars_in_replay - 1)
    to_delete = [*range(blue_end - per_team_to_remove + 1, blue_end + 1)]
    to_delete.extend(range(orange_end - per_team_to_remove + 1 - leftover, orange_end + 1))
    car_data = np.split(data[9:], num_cars_in_replay)
    car_data = np.delete(car_data, to_delete, axis=0)
    car_data = car_data.flatten()
    ball_data = data[:9]
    return np.concatenate((ball_data, car_data))


class ReplaySetter(StateSetter):
    def __init__(self, ndarray_or_file: Union[str, np.ndarray], expand_shrink_cars=False):
        """
        ReplayBasedSetter constructor

        :param ndarray_or_file: A file string or a numpy ndarray of states for a single game mode.
        :param expand_shrink_cars: an integer number of cars to expect, to expand or contract a setter which isn't the right size
        """
        super().__init__()

        self.expand_shrink_cars = expand_shrink_cars
        if isinstance(ndarray_or_file, np.ndarray):
            self.states = ndarray_or_file
        elif isinstance(ndarray_or_file, str):
            self.states = np.load(ndarray_or_file)
        self.probabilities = self.generate_probabilities()

    def generate_probabilities(self):
        """
        Generates probabilities for each state.
        :return: Numpy array of probabilities (summing to 1)
        """
        return np.ones(len(self.states)) / len(self.states)

    @classmethod
    def construct_from_replays(cls, paths_to_replays: List[str], frame_skip: int = 150):
        """
        Alternative constructor that constructs ReplayBasedSetter from replays given as paths.

        :param paths_to_replays: Paths to all the reapls
        :param frame_skip: Every frame_skip frame from the replay will be converted
        :return: Numpy array of frames
        """
        return cls(cls.convert_replays(paths_to_replays, frame_skip))

    @staticmethod
    def convert_replays(paths_to_each_replay: List[str], frame_skip: int = 150, verbose: int = 0, output_location=None):
        from rlgym_tools.replay_converter import convert_replay
        states = []
        for replay in paths_to_each_replay:
            replay_iterator = convert_replay(replay)
            remainder = random.randint(0, frame_skip - 1)  # Vary the delays slightly
            for i, value in enumerate(replay_iterator):
                if i % frame_skip == remainder:
                    game_state, _ = value

                    whole_state = []
                    ball = game_state.ball
                    ball_state = np.concatenate((ball.position, ball.linear_velocity, ball.angular_velocity))

                    whole_state.append(ball_state)
                    for player in game_state.players:
                        whole_state.append(np.concatenate((player.car_data.position,
                                                           player.car_data.euler_angles(),
                                                           player.car_data.linear_velocity,
                                                           player.car_data.angular_velocity,
                                                           np.asarray([player.boost_amount]))))

                    np_state = np.concatenate(whole_state)
                    states.append(np_state)
            if verbose > 0:
                print(replay, "done")

        states = np.asarray(states)
        if output_location is not None:
            np.save(output_location, states)
        return states

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        data = self.states[np.random.choice(len(self.states), p=self.probabilities)]
        if not self.expand_shrink_cars:
            assert len(data) == len(state_wrapper.cars) * 13 + 9, "Data given does not match current game mode"
        elif len(data) < len(state_wrapper.cars) * 13 + 9:
            data = _expand_cars(state_wrapper, data)
        else:
            data = _shrink_cars(state_wrapper, data)
        self._set_ball(state_wrapper, data)
        self._set_cars(state_wrapper, data)

    def _set_cars(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the players according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """

        data = np.split(data[9:], len(state_wrapper.cars))
        for i, car in enumerate(state_wrapper.cars):
            car.set_pos(*data[i][:3])
            car.set_rot(*data[i][3:6])
            car.set_lin_vel(*data[i][6:9])
            car.set_ang_vel(*data[i][9:12])
            car.boost = data[i][12]

    def _set_ball(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the ball according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """
        state_wrapper.ball.set_pos(*data[:3])
        state_wrapper.ball.set_lin_vel(*data[3:6])
        state_wrapper.ball.set_ang_vel(*data[6:9])


