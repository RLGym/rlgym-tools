import math
import warnings
from typing import Sequence

from rlgym_tools.rocket_league.reward_functions.auto_reward_normalizer.running_normalizer import RunningNormalizer


class SimpleZNormalizer(RunningNormalizer):
    def __init__(self, alpha=0.999999, clip=20, center=True, scale=True, ignore_zeros=False):
        self.alpha = alpha
        self.clip = clip
        self.center = center
        self.scale = scale
        self.ignore_zeros = ignore_zeros

        if self.ignore_zeros and self.center:
            warnings.warn(
                "If ignoring zeros, it is recommended that you disable centering "
                "to avoid excessive negative values and keep the sign of the unnormalized reward.",
                UserWarning,
            )

        self._count = 0
        self._sum = 0
        self._sq_sum = 0

    def update(self, values: Sequence):
        # Update the statistics. We do this first to avoid division by zero
        if self.ignore_zeros:
            values = [v for v in values if v != 0]
        c = len(values)
        if c == 0:
            return
        s = sum(values)
        ss = sum(r ** 2 for r in values)

        self._count = self.alpha * self._count + (1 - self.alpha) * c
        self._sum = self.alpha * self._sum + (1 - self.alpha) * s
        self._sq_sum = self.alpha * self._sq_sum + (1 - self.alpha) * ss

    def normalize(self, values: Sequence):
        # Calculate the mean and variance
        if self._count == 0:
            return [math.nan] * len(values)  # The user has to deal with these
        mean = self._sum / self._count if self.center else 0
        var = (self._sq_sum / self._count) - (mean ** 2) if self.scale else 1

        # Normalize the rewards
        std = math.sqrt(var) + 1e-8

        norm_rewards = []
        for reward in values:
            z = (reward - mean) / std
            clipped = min(max(z, -self.clip), self.clip)
            norm_rewards.append(clipped)

        return norm_rewards
