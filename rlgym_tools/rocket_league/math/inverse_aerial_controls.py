import numpy as np
from rlgym.rocket_league.common_values import CAR_MAX_ANG_VEL, TICKS_PER_SECOND

# Code by the legendary Sam Mish, from https://www.smish.dev/rocket_league/inverse_aerial_control/
# NOTE: Modified to produce full-magnitude inputs when at max angvel (thanks Zealan)


T_r = -36.07956616966136  # torque coefficient for roll
T_p = -12.14599781908070  # torque coefficient for pitch
T_y = 8.91962804287785  # torque coefficient for yaw
D_r = -4.47166302201591  # drag coefficient for roll
D_p = -2.798194258050845  # drag coefficient for pitch
D_y = -1.886491900437232  # drag coefficient for yaw


def aerial_inputs(ang_vel_start, ang_vel_end, rot_mat_start, rot_mat_end, dt, is_flipping=False):
    scale = 1.0
    if np.linalg.norm(ang_vel_end) >= CAR_MAX_ANG_VEL - 0.01:
        scale = 1.25  # Scale up so we don't get partial inputs when we hit the max angular velocity
    tau = (ang_vel_end * scale - ang_vel_start) / dt  # net torque in world coordinates
    tst = np.transpose(rot_mat_start)
    # tau1 = np.dot(tst, tau)  # net torque in local coordinates
    tau = np.array([
        tst[0, 0] * tau[0] + tst[0, 1] * tau[1] + tst[0, 2] * tau[2],
        tst[1, 0] * tau[0] + tst[1, 1] * tau[1] + tst[1, 2] * tau[2],
        tst[2, 0] * tau[0] + tst[2, 1] * tau[1] + tst[2, 2] * tau[2]
    ])
    # omega_local1 = np.dot(tst, omega_start)  # beginning-step angular velocity in local coordinates
    ang_vel_local_start = np.array([
        tst[0, 0] * ang_vel_start[0] + tst[0, 1] * ang_vel_start[1] + tst[0, 2] * ang_vel_start[2],
        tst[1, 0] * ang_vel_start[0] + tst[1, 1] * ang_vel_start[1] + tst[1, 2] * ang_vel_start[2],
        tst[2, 0] * ang_vel_start[0] + tst[2, 1] * ang_vel_start[1] + tst[2, 2] * ang_vel_start[2]
    ])

    # assert np.allclose(tau1, tau, equal_nan=True)
    # assert np.allclose(omega_local1, ang_vel_local_start, equal_nan=True)

    rhs = np.array([
        tau[0] - D_r * ang_vel_local_start[0],
        tau[1] - D_p * ang_vel_local_start[1],
        tau[2] - D_y * ang_vel_local_start[2]
    ])

    u = np.array([
        rhs[0] / T_r,  # roll
        rhs[1] / (T_p + np.sign(rhs[1]) * ang_vel_local_start[1] * D_p),  # pitch
        rhs[2] / (T_y - np.sign(rhs[2]) * ang_vel_local_start[2] * D_y)  # yaw
    ])

    # ensure values are between -1 and +1
    u = np.clip(u, -1, +1)

    if is_flipping:
        # From https://github.com/ZealanL/RLCarInputSolver/blob/main/src/AirSolver.cpp
        tst = np.transpose(rot_mat_end)
        ang_vel_local_end = np.array([
            tst[0, 0] * ang_vel_end[0] + tst[0, 1] * ang_vel_end[1] + tst[0, 2] * ang_vel_end[2],
            tst[1, 0] * ang_vel_end[0] + tst[1, 1] * ang_vel_end[1] + tst[1, 2] * ang_vel_end[2],
            tst[2, 0] * ang_vel_end[0] + tst[2, 1] * ang_vel_end[1] + tst[2, 2] * ang_vel_end[2]
        ])

        if abs(ang_vel_local_start[1]) > abs(ang_vel_local_end[1]):
            # Flip cancel
            u[1] = np.sign(ang_vel_local_start[1])

    return u[1], u[2], u[0]  # pitch, yaw, roll
