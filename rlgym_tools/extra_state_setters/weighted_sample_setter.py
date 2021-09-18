from rlgym.utils import RewardFunction
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from random import choices
from typing import Sequence, Union, Tuple


class WeightedSampleSetter(StateSetter):
    """
    Samples StateSetters randomly according to their weights.

    :param state_setters: 1-D array-like of state-setters to be sampled from
    :param weights: 1-D array-like of the weights associated with each entry in state_setters
    """

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]):
        super().__init__()
        self.state_setters = state_setters
        self.weights = weights
        assert len(state_setters) == len(weights), \
            f"Length of state_setters should match the length of weights, " \
            f"instead lengths {len(state_setters)} and {len(weights)} were given respectively."

    @classmethod
    def from_zipped(
            cls,
            *setters_and_weights: Union[StateSetter, Tuple[RewardFunction, float]]
    ) -> "WeightedSampleSetter":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.
        :param setters_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in setters_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, state_wrapper: StateWrapper):
        """
        Executes the reset of randomly sampled state-setter

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        choices(self.state_setters, weights=self.weights)[0].reset(state_wrapper)
