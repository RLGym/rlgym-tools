import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Dict, Optional

import numpy as np
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import ORANGE_TEAM, BLUE_TEAM, TICKS_PER_SECOND, DOUBLEJUMP_MAX_DELAY
from rlgym.rocket_league.math import quat_to_rot_mtx
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator

from rlgym_tools.math.ball import ball_hit_ground
from rlgym_tools.math.inverse_aerial_controls import aerial_inputs
from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardInfo

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
    actions: Dict[int, np.ndarray]
    update_age: Dict[int, float]
    scoreboard: ScoreboardInfo
    episode_seconds_remaining: float
    next_scoring_team: Optional[int]
    winning_team: Optional[int]


def replay_to_rlgym(replay, interpolation: Literal["none", "linear", "rocketsim"] = "rocketsim",
                    predict_pyr=True, calculate_error=False):
    if interpolation not in ("none", "linear", "rocketsim"):
        raise ValueError(f"Interpolation mode {interpolation} not recognized")
    rocketsim_interpolation = interpolation == "rocketsim"
    linear_interpolation = interpolation == "linear"

    players = {p["unique_id"]: p for p in replay.metadata["players"]}
    transition_engine = RocketSimEngine()
    count_orange = sum(p["is_orange"] is True for p in players.values())
    base_mutator = FixedTeamSizeMutator(blue_size=len(players) - count_orange, orange_size=count_orange)
    kickoff_mutator = KickoffMutator()

    average_tick_rate = replay.game_df["delta"].mean() * TICKS_PER_SECOND

    player_ids = sorted(int(k) for k in replay.player_dfs.keys())

    total_pos_error = {}
    total_quat_error = {}
    total_vel_error = {}
    total_ang_vel_error = {}

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
            for uid, player_row in zip(player_ids, car_rows):
                car = state.cars[uid]
                if calculate_error:
                    action, error_pos, error_quat, error_vel, error_ang_vel \
                        = _update_car_and_get_action(car, linear_interpolation, player_row, state,
                                                     calculate_error=True)

                    if frame != start_frame and not np.isnan(error_pos):
                        total_pos_error.setdefault(uid, []).append(error_pos)
                        total_quat_error.setdefault(uid, []).append(error_quat)
                        total_vel_error.setdefault(uid, []).append(error_vel)
                        total_ang_vel_error.setdefault(uid, []).append(error_ang_vel)
                else:
                    action = _update_car_and_get_action(car, linear_interpolation, player_row, state)

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
    if calculate_error:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=4)
        i = 0
        for name, error in [("Position", total_pos_error), ("Quaternion", total_quat_error),
                            ("Velocity", total_vel_error), ("Angular velocity", total_ang_vel_error)]:
            for uid, errors in error.items():
                error[uid] = np.array(errors)
                axs[i].plot(error[uid], label=uid)
            all_errors = np.concatenate(list(error.values()))
            print(f"{name} error: \n"
                  f"\tMean: {np.mean(all_errors)}\n"
                  f"\tStd: {np.std(all_errors)}\n"
                  f"\tMedian: {np.median(all_errors)}\n"
                  f"\tMax: {np.max(all_errors)}")
            # axs[i].hist(all_errors, bins=100)
            # axs[i].set_yscale("log")
            axs[i].set_title(name)
            i += 1
        fig.legend()
        plt.show()


def _prepare_segment_dfs(replay, start_frame, end_frame, linear_interpolation, predict_pyr):
    game_df = replay.game_df.loc[start_frame:end_frame].astype(float)
    ball_df = replay.ball_df.loc[start_frame:end_frame].ffill().fillna(0.).astype(float)
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
                               calculate_error=False):
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

    car.boost_amount = player_row.boost_amount / 100
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
        car.is_flipping = False
        car.on_ground = False
        car.is_jumping = False
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
        car.is_flipping = True
        car.flip_torque = actual_torque
        car.has_flipped = True

        # Pitch/yaw/roll is handled by inverse aerial control function already,
        # it knows about the flip and detects the cancel automatically
        # pitch = yaw = roll = 0
    elif player_row.double_jumped:
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


def get_valid_action_options(car: Car, replay_action: np.ndarray, action_options: np.ndarray, dodge_deadzone=0.5):
    """
    Get the valid action options for a car given a replay action and a set of action options.
    :param car: The car to get the valid actions for
    :param replay_action: The action from the replay
    :param action_options: The action options to choose from
    :param dodge_deadzone: The deadzone for dodges
    :return: A tuple of mask and whether the actions in the mask are optimal
             E.g. is this as good as we could've possibly done, or are there conflicts about what's best.
             If the actions are not always optimal, it might be a sign that the action options
             don't provide good enough coverage.
    """
    optimal = 0
    masks = np.zeros(len(action_options), dtype=bool)

    if car.on_ground or car.can_flip:
        masks += action_options[:, 5] == replay_action[5]
        optimal += 1

    if car.boost_amount > 0:
        masks += action_options[:, 6] == replay_action[6]  # Boost
    optimal += 1

    if replay_action[5] == 1 and not car.on_ground and car.can_flip:
        # Double jump or dodge
        is_dodge = np.abs(replay_action[2:5]).sum() >= dodge_deadzone
        is_dodges = np.abs(action_options[:, 2:5]).sum(axis=1) >= dodge_deadzone
        if is_dodge:
            # Make sure we're flipping in as close to the same direction as possible
            dodge_dir = np.array([replay_action[2], replay_action[3] + replay_action[4]])
            dir_error = ((action_options[:, 2] - dodge_dir[0]) ** 2
                         + (action_options[:, 3:5].sum(axis=1) - dodge_dir[1]) ** 2)
            masks += (dir_error == dir_error.min())
            # And that we're exceeding deadzone
            masks += is_dodges
            optimal += 2
        else:
            # Make sure we're not exceeding deadzone
            masks += ~is_dodges
            optimal += 1
    elif car.on_ground:
        # Prioritize throttle, steer and handbrake
        error = np.abs(action_options[:, 0] - replay_action[0])
        masks += error == error.min()
        error = np.abs(action_options[:, 1] - replay_action[1])
        masks += error == error.min()
        masks += action_options[:, 7] == replay_action[7]
        optimal += 3
    else:
        # Prioritize pitch, yaw and roll
        error = np.linalg.norm(action_options[:, 2:5] - replay_action[2:5]).sum(axis=1)
        masks += error == error.min()
        optimal += 1

    mx = masks.max()
    mask = masks == mx
    is_optimal = mx == optimal
    return mask, is_optimal
