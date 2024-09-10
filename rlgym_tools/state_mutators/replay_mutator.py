import pickle
import random
from typing import Dict, Any, List, Literal

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rlgym_tools.replays.convert import replay_to_rlgym
from rlgym_tools.replays.parsed_replay import ParsedReplay


class ReplayMutator(StateMutator[GameState]):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.states = pickle.load(f)
        self.probabilities = self.assign_probabilities()

    def assign_probabilities(self):
        """
        Assigns probabilities to each state in the replay file.
        Can be overridden to provide custom probabilities.

        :return: A list of probabilities for each state.
        """
        return np.ones(len(self.states)) / len(self.states)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        replay_frame = np.random.choice(self.states, p=self.probabilities)
        new_state = replay_frame.state
        state.tick_count = new_state.tick_count
        state.goal_scored = new_state.goal_scored
        state.config = new_state.config
        state.cars = new_state.cars
        state.ball = new_state.ball
        state.boost_pad_timers = new_state.boost_pad_timers

        shared_info["replay_frame"] = replay_frame

    @staticmethod
    def make_files(replay_files: List[str],
                   output_path: str,
                   frame_skip: int = 30,
                   interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
                   carball_path=None) -> None:
        states = []
        for replay_file in replay_files:
            parsed_replay = ParsedReplay.load(replay_file, carball_path)
            frame = random.randrange(0, frame_skip)  # Randomize starting frame to increase diversity
            for replay_frame in replay_to_rlgym(parsed_replay, interpolation, predict_pyr=True, calculate_error=False):
                if frame % frame_skip == 0:
                    states.append(replay_frame)
                frame += 1
        with open(output_path, "wb") as f:
            pickle.dump(states, f)
