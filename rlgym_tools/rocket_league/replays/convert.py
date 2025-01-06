import warnings
from copy import deepcopy
from typing import Literal, Callable, Iterable

import numpy as np
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import ORANGE_TEAM, BLUE_TEAM, TICKS_PER_SECOND, DOUBLEJUMP_MAX_DELAY, \
    FLIP_TORQUE_TIME
from rlgym.rocket_league.math import quat_to_rot_mtx
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym_tools.rocket_league.math.ball import ball_hit_ground
from rlgym_tools.rocket_league.math.inverse_aerial_controls import aerial_inputs
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo

try:
    import numba
    import scipy  # Not used, but needed for numba

    optional_njit = numba.njit
except ImportError:
    warnings.warn("Numba/scipy not found, falling back to non-jitted functions")
    numba = None


    def optional_njit(f, *args, **kwargs):
        return f


def replay_to_rlgym(
        replay: ParsedReplay,
        interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
        predict_pyr=True,
        calculate_error=False,
        step_rounding: Callable[[float], int] = round,
        modify_action_fn: Callable[[Car, np.ndarray], np.ndarray] = None
) -> Iterable[ReplayFrame]:
    """
    Convert a parsed replay to a sequence of RLGym objects.

    :param replay: The parsed replay to convert
    :param interpolation: The interpolation mode to use. "none" uses the raw data, "linear" interpolates between frames,
                          "rocketsim" uses RocketSim to interpolate between frames.
    :param predict_pyr: Whether to predict pitch, yaw, roll from changes in quaternion
                        (they're the only part of the action not provided by the replay)
    :param calculate_error: Whether to calculate the error between the data from the replay and the interpolated data.
    :param step_rounding: A function to round the number of ticks to step. Default is round.
    :param modify_action_fn: A function to modify the action before it is used. Takes car and action as arguments.
    :return: An iterable of ReplayFrame objects
    """
    if interpolation not in ("none", "linear", "rocketsim"):
        raise ValueError(f"Interpolation mode {interpolation} not recognized")
    rocketsim_interpolation = interpolation == "rocketsim"
    linear_interpolation = interpolation == "linear"

    players = {p["unique_id"]: p for p in replay.metadata["players"]}
    transition_engine = RocketSimEngine(rlbot_delay=False)
    count_orange = sum(p["is_orange"] is True for p in players.values())
    base_mutator = FixedTeamSizeMutator(blue_size=len(players) - count_orange, orange_size=count_orange)
    kickoff_mutator = KickoffMutator()

    hits = set()
    for hit in replay.analyzer.get("hits", []):
        f = hit["frame_number"]
        p = hit["player_unique_id"]
        hits.add((f, int(p)))

    average_tick_rate = replay.game_df["delta"].mean() * TICKS_PER_SECOND

    player_ids = sorted(int(k) for k in replay.player_dfs.keys())

    # Track scoreline
    blue = orange = 0
    final_goal_diff = sum(-1 if goal["is_orange"] else 1 for goal in replay.metadata["game"]["goals"])
    winning_team = None
    if final_goal_diff > 0:
        winning_team = BLUE_TEAM
    elif final_goal_diff < 0:
        winning_team = ORANGE_TEAM

    shared_info = {}
    gameplay_periods = replay.analyzer["gameplay_periods"]

    # For some reason transitioning from regulation to overtime is not always detected properly
    last_period = gameplay_periods[-1]
    start_frame = last_period["start_frame"]
    end_frame = last_period["end_frame"]
    overtime = replay.game_df.loc[start_frame:end_frame]["is_overtime"] == 1
    if overtime.nunique() > 1:
        # The last period is invalid, we need to split it up
        gameplay_periods = gameplay_periods[:-1]
        end_first = overtime.diff().idxmax()
        start_second = (replay.game_df["time"] > replay.game_df.loc[end_first, "time"] + 4).idxmax()
        first_hits = (replay.game_df.loc[start_frame:end_frame]["ball_has_been_hit"] == 1).diff() > 0
        first_hits = first_hits[first_hits].index
        goal_frame = end_frame if abs(replay.ball_df.loc[end_frame, "pos_y"]) > 5000 else None
        gameplay_periods.extend([
            {
                "start_frame": start_frame,
                "first_hit_frame": first_hits[0],
                "goal_frame": None,
                "end_frame": end_first - 1
            },
            {
                "start_frame": start_second,
                "first_hit_frame": first_hits[-1],
                "goal_frame": goal_frame,
                "end_frame": end_frame
            },
        ])

    for g, gameplay_period in enumerate(gameplay_periods):
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
            new_cars[int(uid)] = car
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

        assert ball_tuples[0].pos_x == ball_tuples[0].pos_y == 0

        # prev_actions = {uid: np.zeros(8) for uid in player_ids}

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
            update_age = {}
            errors = {}
            for uid, player_row in zip(player_ids, car_rows):
                car = state.cars[uid]
                hit = (frame, uid) in hits
                if calculate_error:  # and i > 0 and not player_rows[player_ids.index(uid)][i - 1].is_repeat:
                    action, error_pos, error_quat, error_vel, error_ang_vel \
                        = _update_car_and_get_action(car, linear_interpolation, player_row, state,
                                                     calculate_error=True, hit=hit)
                    if frame != start_frame and not np.isnan(error_pos):
                        errors[uid] = {
                            "pos": error_pos,
                            "quat": error_quat,
                            "vel": error_vel,
                            "ang_vel": error_ang_vel,
                        }
                else:
                    action = _update_car_and_get_action(car, linear_interpolation, player_row, state, hit=hit)

                if modify_action_fn is not None:
                    action = modify_action_fn(car, action)

                actions[uid] = action
                update_age[uid] = player_row.update_age if not player_row.is_demoed else 0

            if state.goal_scored:
                if state.scoring_team == BLUE_TEAM:
                    blue += 1
                elif state.scoring_team == ORANGE_TEAM:
                    orange += 1
            scoreboard = ScoreboardInfo(
                game_timer_seconds=game_row.scoreboard_timer,
                kickoff_timer_seconds=game_row.kickoff_timer,
                blue_score=blue,
                orange_score=orange,
                go_to_kickoff=frame == end_frame and g < len(gameplay_periods) - 1,
                is_over=(frame == end_frame
                         and g == len(gameplay_periods) - 1
                         and blue != orange
                         and not (1 < game_row.scoreboard_timer <= 300)
                         and (state.goal_scored
                              or ball_hit_ground(2 * average_tick_rate, state.ball, pre=True)
                              or ball_hit_ground(game_row.delta * TICKS_PER_SECOND, state.ball, pre=False))),
            )

            episode_seconds_remaining = game_row.episode_seconds_remaining

            res = ReplayFrame(
                state=state,
                actions=actions,
                update_age=update_age,
                scoreboard=scoreboard,
                episode_seconds_remaining=episode_seconds_remaining,
                next_scoring_team=next_scoring_team,
                winning_team=winning_team,
            )
            if calculate_error:
                yield res, errors
            else:
                yield res

            if frame == end_frame:
                break

            ticks = step_rounding(game_tuples[i + 1].time * TICKS_PER_SECOND - state.tick_count)
            for uid, action in actions.items():
                repeated_action = action.reshape(1, -1).repeat(ticks, axis=0)
                # repeated_action[:ticks // 2, :] = prev_actions[uid]
                # prev_actions[uid] = action  # Keep the unrepeated action for the next frame
                actions[uid] = repeated_action

            if rocketsim_interpolation:
                transition_engine.set_state(state, {})
                state = transition_engine.step(actions, {})
            else:
                state = deepcopy(state)


def _prepare_segment_dfs(replay, start_frame, end_frame, linear_interpolation, predict_pyr):
    game_df = replay.game_df.loc[start_frame:end_frame].astype(float)
    ball_df = replay.ball_df.loc[start_frame:end_frame].astype(float)
    ball_df["quat_w"] = ball_df["quat_w"].fillna(1.0)
    ball_df = ball_df.ffill().fillna(0.)
    player_dfs = {}
    for uid, pdf in replay.player_dfs.items():
        pdf = pdf.loc[start_frame:end_frame].astype(float)
        physics_cols = ([f"{col}_{axis}" for col in ("pos", "vel", "ang_vel") for axis in "xyz"] +
                        [f"quat_{axis}" for axis in "wxyz"])
        is_repeat = (pdf[physics_cols].diff() == 0).all(axis=1)
        # If something repeats for 4 or more frames, assume it's not a real repeat, just the player standing still
        is_repeat &= (is_repeat.rolling(4).sum() < 4)
        is_demoed = pdf["pos_x"].isna()
        pdf[["throttle", "steer"]] = pdf[["throttle", "steer"]].ffill().fillna(255 / 2)
        pdf = pdf.ffill().fillna(0.)
        pdf["is_repeat"] = is_repeat.astype(float)
        pdf["is_demoed"] = is_demoed.astype(float)
        pdf["got_demoed"] = pdf["is_demoed"].diff() > 0
        pdf["respawned"] = pdf["is_demoed"].diff() < 0
        pdf["jumped"] = pdf["jump_is_active"].fillna(0.).diff() > 0
        pdf["dodged"] = pdf["dodge_is_active"].fillna(0.).diff() > 0
        pdf["double_jumped"] = pdf["double_jump_is_active"].fillna(0.).diff() > 0

        times = game_df["time"].copy()
        times[is_repeat] = np.nan
        times = times.ffill().fillna(0.)
        pdf["update_age"] = game_df["time"] - times

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
        # # Shift the columns used for actions by 1 and add 0s for the first frame
        # action_cols = ["throttle", "steer", "pitch", "yaw", "roll", "jumped", "dodged", "dodge_is_active",
        #                "double_jumped", "flip_car_is_active", "boost_is_active", "handbrake"]
        # pdf[action_cols] = pdf[action_cols].shift(-1).fillna(0.)
        player_dfs[int(uid)] = pdf
    game_df["episode_seconds_remaining"] = game_df["time"].iloc[-1] - game_df["time"]

    # `seconds_remaining` seems to be the ceil of the true game timer
    # We recreate the true timer here
    scoreboard_timer = game_df["seconds_remaining"].copy()
    ball_hit = game_df["ball_has_been_hit"] == 1  # Where the timer is running

    # Synchronize by finding first change in the timer
    decreases = scoreboard_timer[ball_hit].diff() < 0
    if decreases.any():
        first_change = decreases.idxmax()
    else:
        first_change = decreases.index[0]
    delta = game_df["time"][ball_hit] - game_df["time"][ball_hit].loc[first_change]
    scoreboard_timer[ball_hit] = scoreboard_timer.loc[first_change] - delta
    scoreboard_timer[~ball_hit] = np.nan
    scoreboard_timer = scoreboard_timer.ffill().bfill()

    # Handle overtime and negative values
    if "is_overtime" in game_df:
        scoreboard_timer[game_df["is_overtime"] == 1] = np.inf
    scoreboard_timer[scoreboard_timer < 0] = 0

    game_df["scoreboard_timer"] = scoreboard_timer

    # Now kickoff timer
    game_df["kickoff_timer"] = 0.0
    try:
        delta = game_df["time"][~ball_hit] - game_df["time"][~ball_hit].iloc[0]
        kickoff_timer = (5.0 - delta).clip(0, 5)
    except IndexError:
        kickoff_timer = 0.0
    game_df.loc[~ball_hit, "kickoff_timer"] = kickoff_timer

    return ball_df, game_df, player_dfs


def _update_car_and_get_action(car: Car, linear_interpolation: bool, player_row, state: GameState,
                               calculate_error=False, hit: bool = False):
    error_pos = error_quat = error_vel = error_ang_vel = np.nan
    if linear_interpolation or not player_row.is_repeat:
        true_pos = (player_row.pos_x, player_row.pos_y, player_row.pos_z)
        true_vel = (player_row.vel_x, player_row.vel_y, player_row.vel_z)
        true_ang_vel = (player_row.ang_vel_x, player_row.ang_vel_y, player_row.ang_vel_z)
        true_quat = (player_row.quat_w, player_row.quat_x, player_row.quat_y, player_row.quat_z)

        if player_row.got_demoed and not car.is_demoed:
            car.demo_respawn_timer = 3
        elif player_row.is_demoed and not car.is_demoed:
            car.demo_respawn_timer = 1 / TICKS_PER_SECOND
        elif not player_row.is_demoed and car.is_demoed:
            car.demo_respawn_timer = 0

        if calculate_error and not player_row.respawned:
            error_pos = np.linalg.norm(car.physics.position - np.array(true_pos))
            error_quat = min(np.linalg.norm(car.physics.quaternion - np.array(true_quat)),
                             np.linalg.norm(car.physics.quaternion + np.array(true_quat)))
            error_vel = np.linalg.norm(car.physics.linear_velocity - np.array(true_vel))
            error_ang_vel = np.linalg.norm(car.physics.angular_velocity - np.array(true_ang_vel))

        car.physics.position[:] = true_pos
        car.physics.linear_velocity[:] = true_vel
        car.physics.angular_velocity[:] = true_ang_vel
        car.physics.quaternion = np.array(true_quat)  # Uses property setter

    car.boost_amount = player_row.boost_amount
    throttle = 2 * player_row.throttle / 255 - 1
    steer = 2 * player_row.steer / 255 - 1
    if abs(throttle) < 0.01:
        throttle = 0
    if abs(steer) < 0.01:
        steer = 0
    pitch = player_row.pitch
    yaw = player_row.yaw
    roll = player_row.roll
    jump = 0

    if player_row.dodged or player_row.double_jumped:
        # Make sure the car is in a valid state for dodging/double jumping
        car.has_flipped = False
        car.on_ground = False
        # car.jump_time = 0
        car.is_holding_jump = False
        car.has_double_jumped = False
        if car.air_time_since_jump >= DOUBLEJUMP_MAX_DELAY:
            car.air_time_since_jump = DOUBLEJUMP_MAX_DELAY - 1 / TICKS_PER_SECOND
        assert car.can_flip

    if player_row.jumped:
        car.on_ground = True
        car.has_jumped = False
        jump = 1
    elif player_row.dodged:
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
        actual_torque = np.array([player_row.dodge_torque_x, player_row.dodge_torque_y, 0])
        actual_torque = actual_torque / (np.linalg.norm(actual_torque) or 1)

        car.on_ground = False
        # car.is_flipping = True
        car.flip_torque = actual_torque
        car.has_flipped = True
        if car.flip_time >= FLIP_TORQUE_TIME:
            car.flip_time = FLIP_TORQUE_TIME - 1 / TICKS_PER_SECOND

        assert not car.has_flip

        # Pitch/yaw/roll is handled by inverse aerial control function already,
        # it knows about the flip and detects the cancel automatically
        # pitch = yaw = roll = 0
    elif player_row.double_jumped:
        mag = abs(pitch) + abs(yaw) + abs(roll)
        if mag >= state.config.dodge_deadzone:
            # Would not have been a double jump, but a dodge. Correct it
            # {m>d, l*m<d} -> c<d/m for d>0 and m>d
            # New magnitude will be 0.49 with default deadzone
            limiter = 0.98 * state.config.dodge_deadzone / mag
            pitch = pitch * limiter
            roll = roll * limiter
            yaw = yaw * limiter
        jump = 1
    elif player_row.flip_car_is_active and not car.is_autoflipping:
        # I think this is autoflip?
        car.on_ground = False
        jump = 1
    if hit and car.ball_touches == 0:
        car.ball_touches = 1
    boost = player_row.boost_is_active
    handbrake = player_row.handbrake
    action = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
                      dtype=np.float32)
    if calculate_error:
        return action, error_pos, error_quat, error_vel, error_ang_vel
    return action


_numba_quat_2_rot = optional_njit(quat_to_rot_mtx)
_numba_aerial_inputs = optional_njit(aerial_inputs)


@optional_njit
def _multi_aerial_inputs(quats, ang_vels, times, is_flipping):
    pyrs = np.zeros((len(quats), 3), dtype=np.float32)
    next_rot = _numba_quat_2_rot(quats[0])
    for i in range(len(quats) - 1):
        rot = next_rot
        next_rot = _numba_quat_2_rot(quats[i + 1])
        dt = times[i + 1] - times[i]
        pyrs[i, :] = _numba_aerial_inputs(ang_vels[i], ang_vels[i + 1], rot, next_rot, dt, is_flipping[i])
    return pyrs


def pyr_from_dataframe(game_df, player_df):
    is_repeated = player_df["is_repeat"].values == 1
    times = game_df["time"].values.astype(np.float32)
    ang_vels = player_df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values.astype(np.float32)
    quats = player_df[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values.astype(np.float32)
    is_flipping = player_df["dodge_is_active"].values.astype(bool)

    quats = quats[~is_repeated]
    ang_vels = ang_vels[~is_repeated]
    times = times[~is_repeated]
    is_flipping = is_flipping[~is_repeated]

    pyrs = _multi_aerial_inputs(quats, ang_vels, times, is_flipping)

    return pyrs
