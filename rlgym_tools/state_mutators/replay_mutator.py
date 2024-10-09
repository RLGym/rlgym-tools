import random
from typing import Dict, Any, List, Literal

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rlgym_tools.misc.serialize import serialize_game_state, deserialize_game_state, deserialize_scoreboard, \
    serialize_scoreboard
from rlgym_tools.replays.convert import replay_to_rlgym
from rlgym_tools.replays.parsed_replay import ParsedReplay


class ReplayMutator(StateMutator[GameState]):
    def __init__(self, path):
        """
        A state mutator that randomly selects a state from a replay file and applies it to the current state.

        :param path: The path to the collection of replay states.
        """
        with np.load(path) as data:
            self.states = data["states"]
            self.scoreboards = data["scoreboards"] if "scoreboards" in data else None
        self.probabilities = self.assign_probabilities()

    def assign_probabilities(self):
        """
        Assigns probabilities to each state in the replay file.
        Can be overridden to provide custom probabilities.

        :return: A list of probabilities for each state.
        """
        return np.ones(len(self.states)) / len(self.states)

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
                  output_path: str,
                  frame_skip: int = 30,
                  interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
                  carball_path=None) -> None:
        states = []
        scoreboards = []
        for replay_file in replay_files:
            parsed_replay = ParsedReplay.load(replay_file, carball_path)
            frame = random.randrange(0, frame_skip)  # Randomize starting frame to increase diversity
            for replay_frame in replay_to_rlgym(parsed_replay, interpolation, predict_pyr=True, calculate_error=False):
                if frame % frame_skip == 0:
                    serialized_state = serialize_game_state(replay_frame.state)
                    serialized_scoreboard = serialize_scoreboard(replay_frame.scoreboard)
                    states.append(serialized_state)
                    scoreboards.append(serialized_scoreboard)
                frame += 1

        states = np.array(states)
        scoreboards = np.array(scoreboards)

        np.savez_compressed(output_path, states=states, scoreboards=scoreboards)
