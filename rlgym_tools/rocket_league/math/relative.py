from typing import List

import numpy as np
from rlgym.rocket_league.api import PhysicsObject


def relative_physics(origin: PhysicsObject, targets: List[PhysicsObject]) -> List[PhysicsObject]:
    result = []
    rot = origin.rotation_mtx
    for target in targets:
        po = PhysicsObject()
        po.position = (target.position - origin.position) @ rot
        po.linear_velocity = (target.linear_velocity - origin.linear_velocity) @ rot
        po.angular_velocity = target.angular_velocity @ rot
        po.rotation_mtx = target.rotation_mtx @ rot.T
        result.append(po)
    return result


def dodge_relative_rot_mtx(rot_mtx: np.ndarray) -> np.ndarray:
    modified_rot = np.zeros_like(rot_mtx)
    fw = rot_mtx[:2, 0]
    modified_rot[:2, 0] = fw / np.linalg.norm(fw)  # Renormalize forward
    modified_rot[0, 1] = -modified_rot[1, 0]  # Recalculate right vector
    modified_rot[1, 1] = modified_rot[0, 0]  # --||--
    modified_rot[2, 2] = 1  # Set z axis to be up
    return modified_rot


def dodge_relative_physics(origin: PhysicsObject, targets: List[PhysicsObject]) -> List[PhysicsObject]:
    # Dodges only happen in the xy plane, so we pretend the car's hood is facing straight up
    modified_rot = dodge_relative_rot_mtx(origin.rotation_mtx)
    result = []
    for target in targets:
        po = PhysicsObject()
        po.position = (target.position - origin.position) @ modified_rot
        po.linear_velocity = (target.linear_velocity - origin.linear_velocity) @ modified_rot
        po.angular_velocity = target.angular_velocity @ modified_rot
        po.rotation_mtx = target.rotation_mtx @ modified_rot.T
        result.append(po)

    return result
