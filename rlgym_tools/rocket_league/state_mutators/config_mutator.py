from typing import Dict, Any

from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState


class ConfigMutator(StateMutator[GameState]):
    def __init__(self, gravity: float = 1.0, boost_consumption: float = 1.0, dodge_deadzone: float = 0.5):
        self.gravity = gravity
        self.boost_consumption = boost_consumption
        self.dodge_deadzone = dodge_deadzone

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        state.config.gravity = self.gravity
        state.config.boost_consumption = self.boost_consumption
        state.config.dodge_deadzone = self.dodge_deadzone
