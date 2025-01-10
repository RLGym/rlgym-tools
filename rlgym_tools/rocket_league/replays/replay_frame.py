from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo


@dataclass(slots=True)
class ReplayFrame:
    state: GameState  # Current state of the game
    actions: Dict[int, np.ndarray]  # Actions for each player
    update_age: Dict[int, float]  # Time since the replay updated values for each car (>0 means state is interpolated)
    scoreboard: ScoreboardInfo  # Current scoreboard
    episode_seconds_remaining: float  # Time remaining until someone scores or ball hits ground at 0s
    next_scoring_team: Optional[int]  # Team that scores the next goal, None if ball hits ground at 0s
    winning_team: Optional[int]  # Team that wins the game, None if game ended without a winner
