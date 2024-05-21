from typing import Any, Dict, Sequence, Tuple

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState


class WeightedSampleMutator(StateMutator[GameState]):
    def __init__(self, mutators: Sequence[StateMutator], weights: Sequence[float]):
        assert len(mutators) == len(weights)
        self.mutators = mutators
        weights = np.array(weights)
        self.probs = weights / weights.sum()

    @staticmethod
    def from_zipped(*mutator_weights: Tuple[StateMutator, float]):
        mutators, weights = zip(*mutator_weights)
        return WeightedSampleMutator(mutators, weights)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        idx = np.random.choice(len(self.mutators), p=self.probs)
        mutator = self.mutators[idx]
        mutator.apply(state, shared_info)
