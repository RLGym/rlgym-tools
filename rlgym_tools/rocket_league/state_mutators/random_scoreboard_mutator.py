from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.math.skellam import AVERAGE_EPISODE_LENGTH
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo


class RandomScoreboardMutator(StateMutator[GameState]):
    def __init__(self, max_game_length: float = 300.0):
        self.max_game_length = max_game_length

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        scoreboard = shared_info.get("scoreboard", None)
        if scoreboard is None:
            scoreboard = ScoreboardInfo(300, 0, 0, 0, False, False)
            shared_info["scoreboard"] = scoreboard
        else:
            # Indicate that we've set the scoreboard to an initial state
            scoreboard.go_to_kickoff = False
            scoreboard.is_over = False

        num_cars = len(state.cars)
        ael = AVERAGE_EPISODE_LENGTH[num_cars // 2 - 1]  # Average episode length for this gamemode

        # First, check if we should go to overtime
        rate = 0.5 / ael  # Rate of scoring for each team
        b = np.random.poisson(lam=self.max_game_length * rate)
        o = np.random.poisson(lam=self.max_game_length * rate)
        if b == o:
            # We've "played an entire game and ended up equal", so we go to overtime
            scoreboard.game_timer_seconds = float("inf")
        else:
            # No overtime, set to random time between 0 and max_game_length and score based on that
            time_passed = np.random.uniform(0, self.max_game_length)
            b = np.random.poisson(lam=time_passed * rate)
            o = np.random.poisson(lam=time_passed * rate)
            scoreboard.game_timer_seconds = self.max_game_length - time_passed
        scoreboard.blue_score = b
        scoreboard.orange_score = o
