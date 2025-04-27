import matplotlib.pyplot as plt
import time

import numpy as np
from rlgym.rocket_league.common_values import CEILING_Z, SIDE_WALL_X, BACK_WALL_Y

from rlgym_tools.rocket_league.misc.surface import closest_surface, CORNER_BOUNDING_BOX

if __name__ == '__main__':
    # Make a voronoi diagram by finding the closest surface for each position on the field
    z = CEILING_Z * 0.5  # Keep z constant for now
    x = np.linspace(-SIDE_WALL_X, SIDE_WALL_X, 200)
    y = np.linspace(-BACK_WALL_Y, BACK_WALL_Y, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, z)
    positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    t0 = time.perf_counter()
    surfaces = np.array([closest_surface(pos)[0] for pos in positions])
    t1 = time.perf_counter()
    print(t1 - t0)
    surfaces = surfaces.reshape(X.shape).astype(float)
    surfaces[abs(X) + abs(Y) > CORNER_BOUNDING_BOX] = np.nan
    plt.imshow(surfaces, extent=(-SIDE_WALL_X, SIDE_WALL_X, -BACK_WALL_Y, BACK_WALL_Y), origin='lower')
    plt.colorbar()
    plt.title("Closest surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    debug = True
