from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from rlgym.rocket_league.api import GameState

from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardInfo


@dataclass(slots=True)
class ReplayFrame:
    state: GameState
    actions: Dict[int, np.ndarray]
    update_age: Dict[int, float]
    scoreboard: ScoreboardInfo
    episode_seconds_remaining: float
    next_scoring_team: Optional[int]
    winning_team: Optional[int]
