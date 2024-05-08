from typing import Dict, Any

from rlgym.api import StateMutator, StateType
from rlgym.rocket_league.api import GameState


class ReplayMutator(StateMutator[GameState]):
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
