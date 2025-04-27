import math
from enum import IntEnum, auto

import numpy as np
from rlgym.rocket_league.common_values import CEILING_Z, SIDE_WALL_X, BACK_WALL_Y

CORNER_BOUNDING_BOX = 8064  # corners are defined by |x|+|y|=8064


class Surface(IntEnum):
    GROUND = auto()
    CEILING = auto()
    SIDE_WALL_NEG_X = auto()
    SIDE_WALL_POS_X = auto()
    BACK_WALL_NEG_Y = auto()
    BACK_WALL_POS_Y = auto()
    CORNER_NEG_X_NEG_Y = auto()
    CORNER_POS_X_NEG_Y = auto()
    CORNER_NEG_X_POS_Y = auto()
    CORNER_POS_X_POS_Y = auto()

    # OTHER = auto()

    @property
    def is_back_wall(self):
        return self in (self.BACK_WALL_NEG_Y, self.BACK_WALL_POS_Y)

    @property
    def is_side_wall(self):
        return self in (self.SIDE_WALL_NEG_X, self.SIDE_WALL_POS_X)

    def naive_dist_from(self, pos):
        # Calculates "naive" distance to a position.
        # E.g. is uses mathematical definitions of the field bounds
        # It will give accurate distance whenever the current surface is the closest
        x, y, z = pos
        match self:
            # Ground and ceiling are defined by z=0 and z=2044
            case Surface.GROUND:
                dist = z
            case Surface.CEILING:
                dist = CEILING_Z - z

            # Side walls are defined by |x|=5120, so the distance is simply 5120±x
            case Surface.SIDE_WALL_NEG_X:
                dist = SIDE_WALL_X + x
            case Surface.SIDE_WALL_POS_X:
                dist = SIDE_WALL_X - x

            # Back walls are defined by |y|=5120, so the distance is simply 5120±y
            case Surface.BACK_WALL_NEG_Y:
                dist = BACK_WALL_Y + y
            case Surface.BACK_WALL_POS_Y:
                dist = BACK_WALL_Y - y

            # The lines defining the corners are Ax+By+C=0
            # Where A=±1, B=±1, C=8064
            # The distance to the line is given by the formula
            # |Ax+By+C|/sqrt(A^2+B^2)
            # Which simplifies to
            # |±x±y+8064|/sqrt(2)
            # with different signs for the four quadrants
            case Surface.CORNER_NEG_X_NEG_Y:
                dist = abs(x + y + CORNER_BOUNDING_BOX) / math.sqrt(2)
            case Surface.CORNER_POS_X_NEG_Y:
                dist = abs(-x + y + CORNER_BOUNDING_BOX) / math.sqrt(2)
            case Surface.CORNER_NEG_X_POS_Y:
                dist = abs(x - y + CORNER_BOUNDING_BOX) / math.sqrt(2)
            case Surface.CORNER_POS_X_POS_Y:
                dist = abs(-x - y + CORNER_BOUNDING_BOX) / math.sqrt(2)

            # Default to nan
            case _:
                dist = math.nan

        # Abs just to make sure, should only really apply when inside the goal
        return abs(dist)


def closest_surface(pos: np.ndarray) -> (Surface, float):
    closest = min(Surface, key=lambda s: s.naive_dist_from(pos))
    dist = closest.naive_dist_from(pos)
    return closest, dist
