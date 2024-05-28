import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM, OCTANE, BLUE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardProvider, ScoreboardInfo
from rlgym.rocket_league.math import quat_to_rot_mtx
from rlgym_tools.math.inverse_aerial_controls import aerial_inputs
from typing import Literal, Dict, Optional

try:
    import numba
    import scipy  # Not used, but needed for numba

    optional_njit = numba.njit
except ImportError:
    warnings.warn("Numba/scipy not found, falling back to non-jitted functions")
    numba = None


    def optional_njit(f, *args, **kwargs):
        return f


@dataclass(slots=True)
class ReplayFrame:
    state: GameState
    actions: Dict[str, np.ndarray]
    is_latest: Dict[str, bool]
    scoreboard: ScoreboardInfo
    episode_seconds_remaining: float
    next_scoring_team: Optional[int]


def replay_to_rlgym(replay, interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim", predict_pyr=True):
    if interpolation not in ("none", "linear", "rocketsim"):
        raise ValueError(f"Interpolation mode {interpolation} not recognized")
    rocketsim_interpolation = interpolation == "rocketsim"
    linear_interpolation = interpolation == "linear"

    players = {p["unique_id"]: p for p in replay.metadata["players"]}
    transition_engine = RocketSimEngine()
    count_orange = sum(p["is_orange"] is True for p in players.values())
    base_mutator = FixedTeamSizeMutator(blue_size=len(players) - count_orange, orange_size=count_orange)
    kickoff_mutator = KickoffMutator()

    scoreboard_provider = ScoreboardProvider()
    shared_info = {}
    shared_info = scoreboard_provider.create(shared_info)
    scoreboard: ScoreboardInfo = shared_info["scoreboard"]
    scoreboard.game_timer_seconds = replay.game_df["replicated_seconds_remaining"].iloc[[0]].fillna(300.).values[0]
    scoreboard.kickoff_timer_seconds = 5.
    scoreboard.blue_score = 0
    scoreboard.orange_score = 0
    scoreboard.go_to_kickoff = False
    scoreboard.is_over = False

    player_ids = sorted(replay.player_dfs.keys())

    for g, gameplay_period in enumerate(replay.analyzer["gameplay_periods"]):
        start_frame = gameplay_period["start_frame"]
        goal_frame = gameplay_period.get("goal_frame")
        end_frame = goal_frame or gameplay_period["end_frame"]

        # Prepare the starting state
        state = transition_engine.create_base_state()
        base_mutator.apply(state, shared_info)  # Base mutator creates the cars with default values

        # Rename agent ids and set cars to be the correct team
        new_cars = {}
        for uid, car in zip(replay.player_dfs, state.cars.values()):
            player = players[uid]
            # car.hitbox_type = OCTANE  # TODO: Get hitbox from replay
            car.team_num = ORANGE_TEAM if player["is_orange"] else BLUE_TEAM
            new_cars[uid] = car
        state.cars = new_cars

        kickoff_mutator.apply(state, shared_info)
        if rocketsim_interpolation:
            actions = {uid: np.zeros((TICKS_PER_SECOND, 8)) for uid in player_ids}
            transition_engine.set_state(state, shared_info)
            # Step with no actions to let RocketSim calculate internal values
            state = transition_engine.step(actions, shared_info)

        # Prepare the segment dataframes (with some added columns)
        ball_df, game_df, player_dfs = _prepare_segment_dfs(replay, start_frame, end_frame,
                                                            linear_interpolation, predict_pyr)
        next_scoring_team = None
        if goal_frame is not None:
            next_scoring_team = BLUE_TEAM if ball_df["pos_y"].iloc[-1] > 0 else ORANGE_TEAM

        # Prepare tuples for faster iteration
        game_tuples = list(game_df.itertuples())
        ball_tuples = list(ball_df.itertuples())
        player_rows = []
        for pid in player_ids:
            pdf = player_dfs[pid]
            player_rows.append(list(pdf.itertuples()))

        # Iterate over the frames
        for i, game_row, ball_row, car_rows in zip(range(len(game_tuples)),
                                                   game_tuples,
                                                   ball_tuples,
                                                   zip(*player_rows)):
            frame = game_row.Index

            # Set the ball state. Unlike players, it updates every frame
            state.ball.position[:] = (ball_row.pos_x, ball_row.pos_y, ball_row.pos_z)
            state.ball.linear_velocity[:] = (ball_row.vel_x, ball_row.vel_y, ball_row.vel_z)
            state.ball.angular_velocity[:] = (ball_row.ang_vel_x, ball_row.ang_vel_y, ball_row.ang_vel_z)
            state.ball.quaternion = np.array((ball_row.quat_w, ball_row.quat_x,
                                              ball_row.quat_y, ball_row.quat_z))

            state.tick_count = game_row.time * TICKS_PER_SECOND
            state.goal_scored = frame == goal_frame

            actions = {}
            is_latest = {}
            for uid, player_row in zip(player_ids, car_rows):
                car = state.cars[uid]
                action = _update_car_and_get_action(car, linear_interpolation, player_row, state)
                actions[uid] = action
                is_latest[uid] = not player_row.is_repeat

            if i == 0 and g == 0:
                scoreboard_provider.set_state(player_ids, state, shared_info)
            scoreboard_provider.step(player_ids, state, shared_info)
            scoreboard = deepcopy(shared_info["scoreboard"])
            episode_seconds_remaining = game_row.episode_seconds_remaining

            res = ReplayFrame(
                state=state,
                actions=actions,
                is_latest=is_latest,
                scoreboard=scoreboard,
                episode_seconds_remaining=episode_seconds_remaining,
                next_scoring_team=next_scoring_team,
            )
            yield res

            if frame == end_frame:
                break

            ticks = round(game_tuples[i + 1].time * TICKS_PER_SECOND - state.tick_count)
            for uid, action in actions.items():
                actions[uid] = action.reshape(1, -1).repeat(ticks, axis=0)

            if rocketsim_interpolation:
                transition_engine.set_state(state, {})
                state = transition_engine.step(actions, {})
            else:
                state = deepcopy(state)


def _prepare_segment_dfs(replay, start_frame, end_frame, linear_interpolation, predict_pyr):
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

        pyr_cols = ["pitch", "yaw", "roll"]
        pdf[pyr_cols] = np.nan
        if predict_pyr:
            pyrs = pyr_from_dataframe(game_df, pdf)
            pdf.loc[~is_repeat, pyr_cols] = pyrs
        pdf[pyr_cols] = pdf[pyr_cols].ffill().fillna(0.)
        # Player physics info is repeated 1-3 times, so we need to remove those
        if linear_interpolation:
            # Set interpolate cols to NaN if they are repeated
            pdf.loc[is_repeat, physics_cols] = np.nan
            pdf = pdf.interpolate()  # Note that this assumes equal time steps
        player_dfs[uid] = pdf
    game_df["episode_seconds_remaining"] = game_df["time"].iloc[-1] - game_df["time"]
    return ball_df, game_df, player_dfs


def _update_car_and_get_action(car, linear_interpolation, player_row, state):
    if linear_interpolation or not player_row.is_repeat:
        true_pos = (player_row.pos_x, player_row.pos_y, player_row.pos_z)
        true_vel = (player_row.vel_x, player_row.vel_y, player_row.vel_z)
        true_ang_vel = (player_row.ang_vel_x, player_row.ang_vel_y, player_row.ang_vel_z)
        true_quat = (player_row.quat_w, player_row.quat_x, player_row.quat_y, player_row.quat_z)

        car.physics.position[:] = true_pos
        car.physics.linear_velocity[:] = true_vel
        car.physics.angular_velocity[:] = true_ang_vel
        car.physics.quaternion = np.array(true_quat)  # Uses property setter

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
        car.has_flipped = False
        car.is_flipping = False
        car.on_ground = False
        car.is_jumping = False
        car.is_holding_jump = False
        car.has_double_jumped = False
        new_pitch = -player_row.dodge_torque_y
        new_roll = -player_row.dodge_torque_x
        mx = max(abs(new_pitch), abs(new_roll))
        if mx > 0:
            # Project to unit square because why not
            pitch = new_pitch / mx
            yaw = 0
            roll = new_roll / mx
        else:
            # Stall, happens when roll+yaw=0 and abs(roll)+abs(yaw)>deadzone
            roll = 1 if roll > 0 else -1
            yaw = -roll
            pitch = 0
        jump = 1
    elif player_row.dodge_is_active:
        car.on_ground = False
        car.is_flipping = True
        car.flip_torque = np.array([player_row.dodge_torque_x, player_row.dodge_torque_y, 0])
        car.has_flipped = True
        # TODO handle flip cancels
        pitch = yaw = roll = 0
    elif player_row.double_jumped:
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
    elif player_row.flip_car_is_active:
        # I think this is autoflip?
        car.on_ground = False
        jump = 1
    boost = player_row.boost_is_active
    handbrake = player_row.handbrake
    action = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
                      dtype=np.float32)
    return action


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
