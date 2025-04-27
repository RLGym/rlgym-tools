from dataclasses import dataclass

import RocketSim as rsim
import numpy as np
from rlgym.rocket_league.api import Car


@dataclass(frozen=True, slots=True)
class Hitbox:
    length: float
    width: float
    height: float

    offset_x: float
    offset_y: float
    offset_z: float

    front_wheel_radius: float
    front_suspension_rest: float
    front_wheel_offset_x: float
    front_wheel_offset_y: float
    front_wheel_offset_z: float

    back_wheel_radius: float
    back_suspension_rest: float
    back_wheel_offset_x: float
    back_wheel_offset_y: float
    back_wheel_offset_z: float

    @property
    def corner_offsets(self):
        # Center of mass to each corner
        # Order will be:
        # back left bottom, back left top, back right bottom, back right top,
        # front left bottom, front left top, front right bottom, front right top
        mid_length = self.length / 2
        mid_width = self.width / 2
        mid_height = self.height / 2
        naive = np.array([
            [a * mid_length, b * mid_width, c * mid_height]
            for a in (-1, 1)
            for b in (-1, 1)
            for c in (-1, 1)
        ])
        offset = np.array([self.offset_x, self.offset_y, self.offset_z])
        result = naive + offset
        result.setflags(write=False)
        return result

    def corner_positions(self, car: Car):
        physics = car.physics
        position = physics.position
        rot_mtx = physics.rotation_mtx

        return (self.corner_offsets @ rot_mtx.T) + position


OCTANE = Hitbox(
    length=120.507,
    width=86.6994,
    height=38.6591,
    offset_x=13.87566,
    offset_y=0,
    offset_z=20.755,
    front_wheel_radius=12.50,
    front_suspension_rest=38.755,
    front_wheel_offset_x=51.25,
    front_wheel_offset_y=25.90,
    front_wheel_offset_z=20.755,
    back_wheel_radius=15.00,
    back_suspension_rest=37.055,
    back_wheel_offset_x=-33.75,
    back_wheel_offset_y=29.50,
    back_wheel_offset_z=20.755
)

DOMINUS = Hitbox(
    length=130.427,
    width=85.7799,
    height=33.8,
    offset_x=9,
    offset_y=0,
    offset_z=15.75,
    front_wheel_radius=12.00,
    front_suspension_rest=33.95,
    front_wheel_offset_x=50.30,
    front_wheel_offset_y=31.10,
    front_wheel_offset_z=15.75,
    back_wheel_radius=13.50,
    back_suspension_rest=33.85,
    back_wheel_offset_x=-34.75,
    back_wheel_offset_y=33.00,
    back_wheel_offset_z=15.75
)

PLANK = Hitbox(
    length=131.32,
    width=87.1704,
    height=31.8944,
    offset_x=9.00857,
    offset_y=0,
    offset_z=12.0942,
    front_wheel_radius=12.50,
    front_suspension_rest=31.9242,
    front_wheel_offset_x=49.97,
    front_wheel_offset_y=27.80,
    front_wheel_offset_z=12.0942,
    back_wheel_radius=17.00,
    back_suspension_rest=27.9242,
    back_wheel_offset_x=-35.43,
    back_wheel_offset_y=20.28,
    back_wheel_offset_z=12.0942
)

BREAKOUT = Hitbox(
    length=133.992,
    width=83.021,
    height=32.8,
    offset_x=12.5,
    offset_y=0,
    offset_z=11.75,
    front_wheel_radius=13.50,
    front_suspension_rest=29.7,
    front_wheel_offset_x=51.50,
    front_wheel_offset_y=26.67,
    front_wheel_offset_z=11.75,
    back_wheel_radius=15.00,
    back_suspension_rest=29.666,
    back_wheel_offset_x=-35.75,
    back_wheel_offset_y=35.00,
    back_wheel_offset_z=11.75
)

HYBRID = Hitbox(
    length=129.519,
    width=84.6879,
    height=36.6591,
    offset_x=13.8757,
    offset_y=0,
    offset_z=20.755,
    front_wheel_radius=12.50,
    front_suspension_rest=38.755,
    front_wheel_offset_x=51.25,
    front_wheel_offset_y=25.90,
    front_wheel_offset_z=20.755,
    back_wheel_radius=15.00,
    back_suspension_rest=37.055,
    back_wheel_offset_x=-34.00,
    back_wheel_offset_y=29.50,
    back_wheel_offset_z=20.755
)

MERC = Hitbox(
    length=123.22,
    width=79.2103,
    height=44.1591,
    offset_x=11.3757,
    offset_y=0,
    offset_z=21.505,
    front_wheel_radius=15.00,
    front_suspension_rest=39.505,
    front_wheel_offset_x=51.25,
    front_wheel_offset_y=25.90,
    front_wheel_offset_z=21.505,
    back_wheel_radius=15.00,
    back_suspension_rest=39.105,
    back_wheel_offset_x=-33.75,
    back_wheel_offset_y=29.50,
    back_wheel_offset_z=21.505
)

HITBOXES = {
    rsim.CarConfig.OCTANE: OCTANE,
    rsim.CarConfig.DOMINUS: DOMINUS,
    rsim.CarConfig.PLANK: PLANK,
    rsim.CarConfig.BREAKOUT: BREAKOUT,
    rsim.CarConfig.HYBRID: HYBRID,
    rsim.CarConfig.MERC: MERC,
}
