import numpy as np

T_r = -36.07956616966136  # torque coefficient for roll
T_p = -12.14599781908070  # torque coefficient for pitch
T_y = 8.91962804287785  # torque coefficient for yaw
D_r = -4.47166302201591  # drag coefficient for roll
D_p = -2.798194258050845  # drag coefficient for pitch
D_y = -1.886491900437232  # drag coefficient for yaw


def aerial_inputs(omega_start, omega_end, theta_start, dt):
    tau = (omega_end - omega_start) / dt  # net torque in world coordinates
    tst = np.transpose(theta_start)
    # tau1 = np.dot(tst, tau)  # net torque in local coordinates
    tau = np.array([
        tst[0, 0] * tau[0] + tst[0, 1] * tau[1] + tst[0, 2] * tau[2],
        tst[1, 0] * tau[0] + tst[1, 1] * tau[1] + tst[1, 2] * tau[2],
        tst[2, 0] * tau[0] + tst[2, 1] * tau[1] + tst[2, 2] * tau[2]
    ])
    # omega_local1 = np.dot(tst, omega_start)  # beginning-step angular velocity in local coordinates
    omega_local = np.array([
        tst[0, 0] * omega_start[0] + tst[0, 1] * omega_start[1] + tst[0, 2] * omega_start[2],
        tst[1, 0] * omega_start[0] + tst[1, 1] * omega_start[1] + tst[1, 2] * omega_start[2],
        tst[2, 0] * omega_start[0] + tst[2, 1] * omega_start[1] + tst[2, 2] * omega_start[2]
    ])

    # assert np.allclose(tau1, tau, equal_nan=True)
    # assert np.allclose(omega_local1, omega_local, equal_nan=True)

    rhs = np.array([
        tau[0] - D_r * omega_local[0],
        tau[1] - D_p * omega_local[1],
        tau[2] - D_y * omega_local[2]
    ])

    u = np.array([
        rhs[0] / T_r,  # roll
        rhs[1] / (T_p + np.sign(rhs[1]) * omega_local[1] * D_p),  # pitch
        rhs[2] / (T_y - np.sign(rhs[2]) * omega_local[2] * D_y)  # yaw
    ])

    # ensure values are between -1 and +1
    u = np.clip(u, -1, +1)

    return u[1], u[2], u[0]  # pitch, yaw, roll
