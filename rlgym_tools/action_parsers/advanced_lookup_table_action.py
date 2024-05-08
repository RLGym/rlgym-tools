from typing import Any

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction


class AdvancedLookupTableAction(LookupTableAction):
    def __init__(self, throttle_bins: Any = 3,
                 steer_bins: Any = 3,
                 torque_bins: Any = 3,
                 flip_bins: Any = 8,
                 include_stall: bool = False):
        super().__init__()
        self._lookup_table = self.make_lookup_table(throttle_bins, steer_bins, torque_bins, flip_bins, include_stall)

    @staticmethod
    def make_lookup_table(throttle_bins: Any = 3,
                          steer_bins: Any = 3,
                          torque_bins: Any = 3,
                          flip_bins: Any = 8,
                          include_stalls: bool = False):
        # Parse bins
        def parse_bin(b, endpoint=True):
            if isinstance(b, int):
                b = np.linspace(-1, 1, b, endpoint=endpoint)
            else:
                b = np.array(b)
            return b

        throttle_bins = parse_bin(throttle_bins)
        steer_bins = parse_bin(steer_bins)
        torque_bins = parse_bin(torque_bins)
        flip_bins = (parse_bin(flip_bins, endpoint=False) + 1) * np.pi  # Split a circle into equal segments in [0, 2pi)

        actions = []

        # Ground
        pitch = roll = jump = 0
        for throttle in throttle_bins:
            for steer in steer_bins:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        yaw = steer
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Aerial
        jump = handbrake = 0
        for pitch in torque_bins:
            for yaw in torque_bins:
                for roll in torque_bins:
                    if pitch == roll == 0 and np.isclose(yaw, steer_bins).any():
                        continue  # Duplicate with ground
                    magnitude = max(abs(pitch), abs(yaw), abs(roll))
                    if magnitude < 1:
                        continue  # Duplicate rotation direction, only keep max magnitude
                    for boost in (0, 1):
                        throttle = boost
                        steer = yaw
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Flips and jumps
        jump = handbrake = 1  # Enable handbrake for potential wavedashes
        yaw = steer = 0  # Only need roll for sideflip
        angles = [np.nan] + [v for v in flip_bins]
        for angle in angles:
            if np.isnan(angle):
                pitch = roll = 0  # Empty jump
            else:
                pitch = np.sin(angle)
                roll = np.cos(angle)
                # Project to square of diameter 2 because why not
                magnitude = max(abs(pitch), abs(roll))
                pitch /= magnitude
                roll /= magnitude
            for boost in (0, 1):
                throttle = boost
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
        if include_stalls:
            # Add actions for stalling
            actions.append([0, 0, 0, 1, -1, 1, 0, 1])
            actions.append([0, 0, 0, -1, 1, 1, 0, 1])

        actions = np.round(np.array(actions), 3)  # Convert to numpy and remove floating point errors
        assert len(np.unique(actions, axis=0)) == len(actions), 'Duplicate actions found'

        return actions
