import numpy as np
from rlgym.rocket_league.api import Car, PhysicsObject

from rlgym_tools.rocket_league.misc.hitboxes import OCTANE
import matplotlib.pyplot as plt
from rlgym_tools.rocket_league.math.relative import relative_physics

if __name__ == '__main__':
    car = Car()
    car.physics = PhysicsObject()
    car.physics.position = np.array([0, 0, 0])
    car.physics.linear_velocity = np.zeros(3)
    car.physics.angular_velocity = np.zeros(3)
    # car.physics.rotation_mtx = np.eye(3)
    car.physics.euler_angles = np.array([0, np.pi / 8, 0])

    corners = OCTANE.corner_positions(car)
    offsets = OCTANE.corner_offsets
    print(corners)
    print(car.physics.forward)
    print(car.physics.right)

    targets = []
    for corner, offsets in zip(corners, offsets):
        po = PhysicsObject()
        po.position = corner
        po.linear_velocity = np.zeros(3)
        po.angular_velocity = np.zeros(3)
        po.rotation_mtx = np.eye(3)

        print()
        print(corner, offsets)
        print(relative_physics(car.physics, [po])[0].position)

    fig, ax = plt.subplots()
    ax.set_xlim(200, -200)
    ax.set_ylim(-200, 200)
    ax.set_aspect('equal')
    ax.set_title('Car corners')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()
    ax.scatter(corners[:, 0], corners[:, 1], c='r', label='Corners')

    plt.show()
