from typing import Dict, Any, Union

from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import OCTANE, DOMINUS, PLANK, BREAKOUT, HYBRID, MERC


class HitboxMutator(StateMutator[GameState]):
    """
    Sets the hitbox of all cars in the game to the specified hitbox type.
    """

    def __init__(self, hitbox_type: Union[int, str]):
        if isinstance(hitbox_type, str):
            hitbox_type = {
                "octane": OCTANE,
                "dominus": DOMINUS,
                "plank": PLANK,
                "breakout": BREAKOUT,
                "hybrid": HYBRID,
                "merc": MERC,
            }[hitbox_type.lower()]
        assert hitbox_type in (OCTANE, DOMINUS, PLANK, BREAKOUT, HYBRID, MERC), "Invalid hitbox"
        self.hitbox_type = hitbox_type

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        for car in state.cars.values():
            car.hitbox_type = self.hitbox_type
