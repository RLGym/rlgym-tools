import os.path
import random
from typing import Dict, Any, List, Literal, Tuple, Union, Optional

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rlgym_tools.misc.serialize import serialize_game_state, deserialize_game_state, deserialize_scoreboard, \
    serialize_scoreboard, GS_CARS, GS_CAR_LENGTH
from rlgym_tools.replays.convert import replay_to_rlgym
from rlgym_tools.replays.parsed_replay import ParsedReplay


class ReplayMutator(StateMutator[GameState]):
    def __init__(
            self,
            path_or_arrays: Union[str, dict[str, np.ndarray]],
    ):
        """
        A state mutator that randomly selects a state from a replay file and applies it to the current state.

        :param path_or_arrays: Path to npz file or directory containing replay data, or a dictionary containing the
                               arrays directly.
        """

        scoreboards = None
        probabilities = None

        if isinstance(path_or_arrays, dict):
            arrays = path_or_arrays
            states = arrays["states"]
            scoreboards = arrays.get("scoreboards")
            probabilities = arrays.get("probabilities")
        else:
            path = path_or_arrays
            if os.path.isdir(path):
                states = np.memmap(os.path.join(path, "states.dat"), dtype='float32', mode='r')
                if os.path.isfile((fpath := os.path.join(path, "scoreboards.dat"))):
                    scoreboards = np.memmap(fpath, dtype='float32', mode='r')
                if os.path.isfile((fpath := os.path.join(path, "probabilities.npy"))):
                    probabilities = np.load(fpath)
            else:
                with np.load(path) as data:
                    states = data["states"]
                    scoreboards = data.get("scoreboards")
                    probabilities = data.get("probabilities")

        self.states = states
        self.scoreboards = scoreboards
        self.probabilities = self.assign_probabilities() if probabilities is None else probabilities

    def assign_probabilities(self):
        """
        Assigns probabilities to each state in the replay file.
        Can be overridden to provide custom probabilities based on the states.

        :return: A list of probabilities for each state.
        """
        probs = np.full(shape=len(self.states), fill_value=1 / len(self.states), dtype=np.float32)
        return probs

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        idx = np.random.choice(len(self.states), p=self.probabilities)
        new_state = deserialize_game_state(self.states[idx])
        state.tick_count = new_state.tick_count
        state.goal_scored = new_state.goal_scored
        state.config = new_state.config
        state.cars = new_state.cars
        state.ball = new_state.ball
        state.boost_pad_timers = new_state.boost_pad_timers

        if self.scoreboards is not None:
            shared_info["scoreboard"] = deserialize_scoreboard(self.scoreboards[idx])

    @staticmethod
    def make_file(replay_files: List[str],
                  output_path: Optional[str],
                  do_memory_map: bool = False,
                  frame_skip: int = 30,
                  interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
                  carball_path=None,
                  max_num_players=6) -> Tuple[np.ndarray, np.ndarray]:
        size = len(replay_files) * 5 * 60 * 30 // frame_skip  # Approximate size of replay files

        states_shape = (size, GS_CARS.start + max_num_players * GS_CAR_LENGTH)

        if do_memory_map:
            assert os.path.isdir(output_path), "Output path must be a directory"
            states = np.memmap(os.path.join(output_path, "states.dat"),
                               dtype='float32', mode='w+', shape=states_shape)
            scoreboards = np.memmap(os.path.join(output_path, "scoreboards.dat"),
                                    dtype='float32', mode='w+', shape=(size, 6))
        else:
            states = np.zeros(states_shape, dtype=np.float32)
            scoreboards = np.zeros((size, 6), dtype=np.float32)

        most_cars = 0

        i = 0
        for k, replay_file in enumerate(replay_files):
            parsed_replay = ParsedReplay.load(replay_file, carball_path)
            frame = random.randrange(0, frame_skip)  # Randomize starting frame to increase diversity
            for replay_frame in replay_to_rlgym(parsed_replay, interpolation, predict_pyr=True, calculate_error=False):
                if frame % frame_skip == 0:
                    serialized_state = serialize_game_state(replay_frame.state)
                    serialized_scoreboard = serialize_scoreboard(replay_frame.scoreboard)

                    if i >= size:
                        # Make a new estimate of the size of the replay files and expand the arrays
                        frames_per_replay = (i + 1) / (k or 1)
                        new_size = (len(replay_files) + 1) * frames_per_replay  # +1 so we don't have to resize a lot
                        states.resize((new_size, states.shape[1]))
                        scoreboards.resize((new_size, 6))

                    if len(replay_frame.state.cars) > most_cars:
                        most_cars = len(replay_frame.state.cars)

                    states[i] = serialized_state
                    scoreboards[i] = serialized_scoreboard

                    i += 1
                frame += 1

        states.resize((i, states.shape[1]))
        scoreboards.resize((i, 6))

        if not do_memory_map:
            np.savez_compressed(output_path, states=states, scoreboards=scoreboards)

        return states, scoreboards
