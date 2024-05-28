import numpy as np
from rlgym.rocket_league.common_values import ORANGE_TEAM, OCTANE, BLUE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardProvider, ScoreboardInfo
from rlgym.rocket_league.math import quat_to_rot_mtx
from rlgym_tools.math.inverse_aerial_controls import aerial_inputs

try:
    raise ImportError
    import numba
    import scipy  # Not used, but needed for numba

    optional_njit = numba.njit
except ImportError:
    numba = None


    def optional_njit(f, *args, **kwargs):
        return f


def replay_to_rlgym(replay, rocketsim_interpolation=True, debug=False):
    players = {p["unique_id"]: p for p in replay.metadata["players"]}
    transition_engine = RocketSimEngine()
    count_orange = sum(p["is_orange"] is True for p in players.values())
    base_mutator = FixedTeamSizeMutator(blue_size=len(players) - count_orange, orange_size=count_orange)
    kickoff_mutator = KickoffMutator()

    scoreboard_provider = ScoreboardProvider()
    shared_info = {}
    shared_info = scoreboard_provider.create(shared_info)
    scoreboard: ScoreboardInfo = shared_info["scoreboard"]
    scoreboard.game_timer_seconds = 300.
    scoreboard.kickoff_timer_seconds = 5.
    scoreboard.blue_score = 0
    scoreboard.orange_score = 0
    scoreboard.go_to_kickoff = True
    scoreboard.is_over = False

    for g, gameplay_period in enumerate(replay.analyzer["gameplay_periods"]):
        start_frame = gameplay_period["start_frame"]
        goal_frame = gameplay_period.get("goal_frame")
        end_frame = goal_frame or gameplay_period["end_frame"]

        base_state = transition_engine.create_base_state()

        base_mutator.apply(base_state, shared_info)  # Let base mutator create the cars
        cars = list(base_state.cars.values())

        new_cars = {}
        actions = {}
        for uid, car in zip(replay.player_dfs, cars):
            player = players[uid]
            car.hitbox_type = OCTANE  # TODO: Get hitbox from replay
            car.team_num = ORANGE_TEAM if player["is_orange"] else BLUE_TEAM
            new_cars[uid] = car
            actions[uid] = np.zeros((120, 8), dtype=np.float32)
        base_state.cars = new_cars

        kickoff_mutator.apply(base_state, shared_info)
        transition_engine.set_state(base_state, shared_info)
        # Step with no actions to let RocketSim calculate internal values
        state = transition_engine.step(actions, shared_info)

        game_df = replay.game_df.loc[start_frame:end_frame].astype(float)
        ball_df = replay.ball_df.loc[start_frame:end_frame].ffill().fillna(0.).astype(float)
        player_dfs = {}
        for uid, pdf in replay.player_dfs.items():
            pdf = pdf.loc[start_frame:end_frame].astype(float)
            physics_cols = ([f"{col}_{axis}" for col in ("pos", "vel", "ang_vel") for axis in "xyz"] +
                            [f"quat_{axis}" for axis in "wxyz"])
            is_repeat = (pdf[physics_cols].diff() == 0).all(axis=1)
            is_demoed = pdf["pos_x"].isna()
            pdf = pdf.ffill().fillna(0.)
            pdf["is_repeat"] = is_repeat.astype(float)
            pdf["is_demoed"] = is_demoed.astype(float)
            pdf["jumped"] = pdf["jump_is_active"].fillna(0.).diff() > 0
            pdf["dodged"] = pdf["dodge_is_active"].fillna(0.).diff() > 0
            pdf["double_jumped"] = pdf["double_jump_is_active"].fillna(0.).diff() > 0

            pyrs = pyr_from_dataframe(game_df, pdf)
            pdf[["pitch", "yaw", "roll"]] = np.nan
            pdf.loc[~is_repeat, ["pitch", "yaw", "roll"]] = pyrs
            pdf[["pitch", "yaw", "roll"]] = pdf[["pitch", "yaw", "roll"]].ffill().fillna(0.)
            # Player physics info is repeated 1-3 times, so we need to remove those
            if interpolate:
                # Set interpolate cols to NaN if they are repeated
                pdf.loc[is_repeat, physics_cols] = np.nan
                pdf = pdf.interpolate()  # Note that this assumes equal time steps
            player_dfs[uid] = pdf

        next_scoring_team = None
        if goal_frame is not None:
            next_scoring_team = BLUE_TEAM if ball_df["pos_y"].iloc[-1] > 0 else ORANGE_TEAM
        game_df["episode_seconds_remaining"] = game_df["time"].iloc[-1] - game_df["time"]

        game_tuples = list(game_df.itertuples())
        ball_tuples = list(ball_df.itertuples())
        player_ids = sorted(player_dfs.keys())
        player_rows = []
        for pid in player_ids:
            pdf = player_dfs[pid]
            player_rows.append(list(pdf.itertuples()))

        for i, game_row, ball_row, car_rows in zip(range(len(game_tuples)),
                                                   game_tuples,
                                                   ball_tuples,
                                                   zip(*player_rows)):
            frame = game_row.Index
            state.ball.position[:] = (ball_row.pos_x, ball_row.pos_y, ball_row.pos_z)
            state.ball.linear_velocity[:] = (ball_row.vel_x, ball_row.vel_y, ball_row.vel_z)
            state.ball.angular_velocity[:] = (ball_row.ang_vel_x, ball_row.ang_vel_y, ball_row.ang_vel_z)
            state.ball.quaternion = np.array((ball_row.quat_w, ball_row.quat_x,
                                              ball_row.quat_y, ball_row.quat_z))

            state.tick_count = game_row.time * TICKS_PER_SECOND
            state.goal_scored = frame == goal_frame

            actions = {}
            is_latest = {}
            for j, uid, player_row in zip(range(len(car_rows)), player_ids, car_rows):
                car = state.cars[uid]
                if interpolate or not player_row.is_repeat:
                    if debug:
                        pred_pos = car.physics.position.copy()
                        pred_vel = car.physics.linear_velocity.copy()
                        pred_ang_vel = car.physics.angular_velocity.copy()
                        pred_quat = car.physics.quaternion.copy()

                    true_pos = (player_row.pos_x, player_row.pos_y, player_row.pos_z)
                    true_vel = (player_row.vel_x, player_row.vel_y, player_row.vel_z)
                    true_ang_vel = (player_row.ang_vel_x, player_row.ang_vel_y, player_row.ang_vel_z)
                    true_quat = (player_row.quat_w, player_row.quat_x, player_row.quat_y, player_row.quat_z)

                    car.physics.position[:] = true_pos
                    car.physics.linear_velocity[:] = true_vel
                    car.physics.angular_velocity[:] = true_ang_vel
                    car.physics.quaternion = np.array(true_quat)  # Uses property setter
                    if debug:
                        print(f"Set physics for player {uid} at frame {frame}")

                    if debug:
                        diff_pos = np.linalg.norm(pred_pos - true_pos)
                        diff_vel = np.linalg.norm(pred_vel - true_vel)
                        diff_ang_vel = np.linalg.norm(pred_ang_vel - true_ang_vel)
                        diff_quat = min(np.linalg.norm(pred_quat - true_quat),
                                        np.linalg.norm(pred_quat + true_quat))
                        print(f"Diff pos: {diff_pos:.2f}, vel: {diff_vel:.2f}, ang_vel: {diff_ang_vel:.2f}, "
                              f"quat: {diff_quat:.2f} at frame {frame} for player {uid}")

                    if car.demo_respawn_timer == 0 and player_row.is_demoed:
                        car.demo_respawn_timer = 3
                car.boost_amount = player_row.boost_amount / 100

                throttle = 2 * player_row.throttle / 255 - 1
                steer = 2 * player_row.steer / 255 - 1
                pitch = player_row.pitch
                yaw = player_row.yaw
                roll = player_row.roll
                jump = 0
                if player_row.jumped:
                    car.on_ground = True
                    car.has_jumped = False
                    jump = 1
                elif player_row.dodged:
                    if debug:
                        dodge_torque = np.array([player_row.dodge_torque_x, player_row.dodge_torque_y, 0])
                        dodge_torque = dodge_torque / np.linalg.norm(dodge_torque)
                        print(f"Player {uid} dodged at frame {frame} ({dodge_torque})")
                    car.has_flipped = False
                    car.is_flipping = False
                    car.on_ground = False
                    car.is_jumping = False
                    car.is_holding_jump = False
                    car.has_double_jumped = False
                    # TODO find out why it's not dodging in RLViser
                    pitch = -player_row.dodge_torque_y
                    yaw = 0
                    roll = -player_row.dodge_torque_x
                    mx = max(abs(pitch), abs(roll))
                    if mx > 0:
                        # Project to unit square because why not
                        pitch /= mx
                        roll /= mx
                    else:
                        # Stall, happens when roll+yaw=0 and abs(roll)+abs(yaw)>deadzone
                        pitch = 0
                        yaw = -1
                        roll = 1
                    jump = 1
                elif player_row.dodge_is_active:
                    if debug:
                        print(f"Player {uid} should be dodging at frame {frame} ({car.flip_torque}) "
                              f"{car.is_flipping=}, {car.physics.up=}")
                    car.on_ground = False
                    car.is_flipping = True
                    car.flip_torque = np.array([player_row.dodge_torque_x, player_row.dodge_torque_y, 0])
                    car.has_flipped = True
                    # TODO handle flip cancels
                    pitch = yaw = roll = 0
                elif player_row.double_jumped:
                    if debug:
                        print(f"Player {uid} double jumped at frame {frame}")
                    car.on_ground = False
                    car.is_jumping = False
                    car.is_holding_jump = False
                    car.has_flipped = False
                    car.has_double_jumped = False
                    mag = abs(pitch) + abs(yaw) + abs(roll)
                    if mag >= state.config.dodge_deadzone:
                        # Would not have been a double jump, but a dodge. Correct it
                        limiter = 0.98 * state.config.dodge_deadzone / mag
                        pitch = pitch * limiter
                        roll = roll * limiter
                        yaw = yaw * limiter
                    jump = 1

                if debug and not car.on_ground and car.can_flip and not car.has_jumped:
                    print(f"Infinite flip? Player {uid} at frame {frame}")

                boost = player_row.boost_is_active
                handbrake = player_row.handbrake
                action = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
                                  dtype=np.float32)
                actions[uid] = action
                is_latest[uid] = not player_row.is_repeat

            if i == 0 and g == 0:
                scoreboard_provider.set_state(player_ids, state, shared_info)
            scoreboard_provider.step(player_ids, state, shared_info)
            scoreboard = shared_info["scoreboard"]
            episode_seconds_remaining = game_row.episode_seconds_remaining

            yield state, actions, is_latest, scoreboard, episode_seconds_remaining, next_scoring_team

            if frame == end_frame:
                break

            ticks = round(game_tuples[i + 1].time * TICKS_PER_SECOND - state.tick_count)
            for uid, action in actions.items():
                actions[uid] = action.reshape(1, -1).repeat(ticks, axis=0)
            transition_engine.set_state(state, {})
            state = transition_engine.step(actions, {})


_numba_quat_2_rot = optional_njit(quat_to_rot_mtx)
_numba_aerial_inputs = optional_njit(aerial_inputs)


@optional_njit
def _multi_aerial_inputs(quats, ang_vels, times):
    pyrs = np.zeros((len(quats), 3), dtype=np.float32)
    for i in range(len(quats) - 1):
        rot = _numba_quat_2_rot(quats[i])
        dt = times[i + 1] - times[i]
        pyrs[i, :] = _numba_aerial_inputs(ang_vels[i], ang_vels[i + 1], rot, dt)
    return pyrs


def pyr_from_dataframe(game_df, player_df):
    is_repeated = player_df["is_repeat"].values == 1
    times = game_df["time"].values.astype(np.float32)
    ang_vels = player_df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values.astype(np.float32)
    quats = player_df[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values.astype(np.float32)

    quats = quats[~is_repeated]
    ang_vels = ang_vels[~is_repeated]
    times = times[~is_repeated]

    pyrs = _multi_aerial_inputs(quats, ang_vels, times)

    return pyrs
