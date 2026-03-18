import contextlib
import logging
import os.path
import random
import tempfile
from typing import Dict, Any, List, Literal, Union, Optional

import numpy as np
from numpy.lib.format import open_memmap
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.misc.serialize import GS_CARS, GS_CAR_LENGTH, RF_AGENT_IDS_START, serialize_replay_frame, \
    deserialize_replay_frame, RF_ACTION_SIZE
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame


class ReplayMutator(StateMutator[GameState]):
    def __init__(
            self,
            replay_frames: Union[str, np.ndarray],
            probabilities: Union[None, str, np.ndarray] = None,
    ):
        """
        A state mutator that randomly selects a state from a replay file and applies it to the current state.

        High-level replay states for 1s, 2s, and 3s are available here:
        https://drive.google.com/drive/folders/1T3Fu04t2wV6kmTVUwK9vyKgFGh-hqVCJ?usp=sharing

        :param replay_frames: The replay file to sample states from. Can be a .npz file, a .npy file, or a numpy array.
        :param probabilities: The probabilities of selecting each state in the replay file. Defaults to uniform.
        """

        if isinstance(replay_frames, str):
            if replay_frames.endswith(".npz"):
                with np.load(replay_frames) as data:
                    replay_frames = data["replay_frames"]
                    if probabilities is None:
                        probabilities = data.get("probabilities")
            elif replay_frames.endswith(".npy"):
                replay_frames = open_memmap(replay_frames, dtype='float32', mode='r')
            else:
                raise ValueError("Invalid replay file")

        if isinstance(probabilities, str):
            probabilities = np.load(probabilities)  # Assume .npy file

        self.replay_frames = replay_frames
        self.probabilities = self.assign_probabilities() if probabilities is None else probabilities

    def assign_probabilities(self):
        """
        Assigns probabilities to each state in the replay file.
        Can be overridden to provide custom probabilities based on the states.

        :return: A list of probabilities for each state.
        """
        return None  # Uniform distribution

    def __getitem__(self, item) -> ReplayFrame:
        return deserialize_replay_frame(self.replay_frames[item])

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        idx = np.random.choice(len(self.replay_frames), p=self.probabilities)
        replay_frame = self[idx]
        new_state = replay_frame.state
        state.tick_count = round(new_state.tick_count)
        state.goal_scored = new_state.goal_scored
        state.config = new_state.config
        state.cars = new_state.cars
        state.ball = new_state.ball
        state.boost_pad_timers = new_state.boost_pad_timers

        shared_info["replay_frame"] = replay_frame  # In case anyone needs more than state and scoreboard
        shared_info["scoreboard"] = replay_frame.scoreboard

    @staticmethod
    def make_file(replay_files: List[str],
                  output_path: Optional[str],
                  do_memory_map: bool = False,
                  frame_skip: int = 30,
                  interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
                  carball_path=None,
                  max_num_players=6) -> np.ndarray:
        size = len(replay_files) * 5 * 60 * 30 // frame_skip  # Initial guess of frame count

        max_state_size = GS_CARS.start + max_num_players * GS_CAR_LENGTH
        max_replay_frame_size = (RF_AGENT_IDS_START
                                 + 2 * max_num_players  # agent_id, update_age
                                 + max_num_players * RF_ACTION_SIZE  # actions
                                 + max_state_size)

        if do_memory_map:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, "replay_frames.npy")
            # Use a temporary raw memmap file for building, then copy to a proper .npy file at the end
            temp_dir = tempfile.TemporaryDirectory()
        else:
            temp_dir = contextlib.nullcontext()

        with temp_dir:
            if do_memory_map:
                temp_path = os.path.join(temp_dir.name, "temp.dat")
                replay_frames = np.memmap(temp_path, dtype='float32', mode='w+',
                                          shape=(size, max_replay_frame_size))
            else:
                replay_frames = np.zeros((size, max_replay_frame_size), dtype=np.float32)

            i = 0
            for k, replay_file in enumerate(replay_files):
                try:
                    parsed_replay = ParsedReplay.load(replay_file, carball_path)
                    frame = random.randrange(0, frame_skip)  # Randomize starting frame to increase diversity
                    for replay_frame in replay_to_rlgym(parsed_replay, interpolation,
                                                        predict_pyr=True, calculate_error=False):
                        if frame % frame_skip == 0:
                            serialized_frame = serialize_replay_frame(replay_frame)
                            if len(serialized_frame) > max_replay_frame_size:
                                frame += 1
                                continue

                            if i >= size:
                                # Make a new estimate of the size of the replay files and expand the arrays
                                frames_per_replay = (i + 1) / (k or 1)
                                new_size = (
                                                   len(replay_files) + 1) * frames_per_replay  # +1 so we don't have to resize a lot
                                new_size = round(new_size)
                                if do_memory_map:
                                    # Flush and re-open the raw memmap with a larger size
                                    replay_frames.flush()
                                    del replay_frames
                                    replay_frames = np.memmap(
                                        temp_path, dtype='float32', mode='r+',
                                        shape=(new_size, max_replay_frame_size),
                                    )
                                else:
                                    # Shouldn't need refcheck, just makes it fail in debug mode
                                    replay_frames.resize((new_size, max_replay_frame_size), refcheck=False)
                                size = new_size

                            replay_frames[i, :len(serialized_frame)] = serialized_frame

                            i += 1
                        frame += 1
                except Exception as e:
                    logging.warning("Failed to process replay file %s: %s", replay_file, e)
                    continue

            if do_memory_map:
                # Create the final .npy file with only the relevant rows
                replay_frames.flush()
                del replay_frames
                temp_frames = np.memmap(temp_path, dtype='float32', mode='r',
                                        shape=(i, max_replay_frame_size))
                final_frames = open_memmap(output_path, dtype='float32', mode='w+',
                                           shape=(i, max_replay_frame_size))
                final_frames[:] = temp_frames[:]
                final_frames.flush()
                del temp_frames
                replay_frames = open_memmap(output_path, dtype='float32', mode='r')
            else:
                replay_frames.resize((i, max_replay_frame_size), refcheck=False)
                if output_path is not None:
                    np.savez_compressed(output_path, replay_frames=replay_frames)

        return replay_frames
