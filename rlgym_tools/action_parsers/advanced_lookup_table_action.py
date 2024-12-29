from typing import Any

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction


def _parse_bin(b, endpoint=True):
    if isinstance(b, int):
        b = np.linspace(-1, 1, b, endpoint=endpoint)
    else:
        b = np.array(b)
    return b


def _subdivide(lo, hi, depth=0):
    # Add points to a grid of size [lo, hi] x [lo, hi]
    # alternating between square and diamond steps as in the diamond-square algorithm
    # Basically, if we have:
    # •   •   •
    #
    # •   •   •
    #
    # •   •   •
    # then we add points to the grid like this:
    # •   •   •
    #   •   •
    # •   •   •
    #   •   •
    # •   •   •
    # and then like this:
    # • • • • •
    # • • • • •
    # • • • • •
    # • • • • •
    # • • • • •
    # instead of going straight from the first grid to the third grid
    if depth < 0:
        return
    # Square step
    count = 1 + 2 ** (depth // 2)
    bins, delta = np.linspace(lo, hi, count, retstep=True)
    for nx in bins:
        for ny in bins:
            yield nx, ny
    # Diamond step, just shift the square step diagonally and ignore those that exceed hi
    if depth % 2 == 1:
        bins = bins[:-1] + delta / 2
        for nx in bins:
            for ny in bins:
                yield nx, ny


class AdvancedLookupTableAction(LookupTableAction):
    def __init__(self, throttle_bins: Any = 3,
                 steer_bins: Any = 3,
                 torque_subdivisions: Any = 2,
                 flip_bins: Any = 8,
                 include_stalls: bool = False):
        super().__init__()
        self._lookup_table = self.make_lookup_table(throttle_bins, steer_bins, torque_subdivisions, flip_bins,
                                                    include_stalls)

    @staticmethod
    def make_lookup_table(throttle_bins: Any = 3,
                          steer_bins: Any = 3,
                          torque_subdivisions: Any = 2,
                          flip_bins: Any = 8,
                          include_stalls: bool = False):
        # Parse bins
        throttle_bins = _parse_bin(throttle_bins)
        steer_bins = _parse_bin(steer_bins)
        flip_bins = (_parse_bin(flip_bins,
                                endpoint=False) + 1) * np.pi  # Split a circle into equal segments in [0, 2pi)
        if isinstance(torque_subdivisions, int):
            torque_face = np.array([
                [x, y]
                for x, y in _subdivide(-1, 1, torque_subdivisions)
            ])
        else:
            if isinstance(torque_subdivisions, np.ndarray) and torque_subdivisions.ndim == 2:
                torque_face = torque_subdivisions
            else:
                torque_subdivisions = _parse_bin(torque_subdivisions)
                torque_face = np.array([
                    [x, y]
                    for x in torque_subdivisions
                    for y in torque_subdivisions
                ])

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
        points = np.array([
            np.insert(p, i, side)
            for i in range(3)  # Determines which axis we select faces from
            for side in (-1, 1)  # Determines which side we select
            for p in torque_face  # Selects where we are on the face
        ])
        points = np.unique(points, axis=0)  # Remove duplicates (corners and edges of the cube)
        for p in points:
            pitch, yaw, roll = p.tolist()
            if pitch == roll == 0 and np.isclose(yaw, steer_bins).any():
                continue  # Duplicate with ground
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
