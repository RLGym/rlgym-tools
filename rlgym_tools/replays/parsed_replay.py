import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from rlgym.rocket_league.api import GameConfig
from rlgym.rocket_league.common_values import TICKS_PER_SECOND, OCTANE, ORANGE_TEAM, BLUE_TEAM
from rlgym.rocket_league.math import quat_to_rot_mtx
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

from rlgym_tools.math.inverse_aerial_controls import aerial_inputs

try:
    raise ImportError
    import numba
    import scipy

    optional_njit = numba.njit
except ImportError:
    numba = None


    def optional_njit(f, *args, **kwargs):
        return f

CARBALL_COMMAND = '{} -i "{}" -o "{}" parquet'

ENV = os.environ.copy()
ENV["NO_COLOR"] = "1"


def process_replay(replay_path, output_folder, carball_path=None, skip_existing=True):
    if carball_path is None:
        # Use carball.exe in the same directory as this script
        carball_path = os.path.join(os.path.dirname(__file__), "carball.exe")
    folder, fn = os.path.split(replay_path)
    replay_name = fn.replace(".replay", "")
    processed_folder = os.path.join(output_folder, replay_name)
    if os.path.isdir(processed_folder) and len(os.listdir(processed_folder)) > 0:
        if skip_existing:
            return
        else:
            os.rmdir(processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    with open(os.path.join(processed_folder, "carball.o.log"), "w", encoding="utf8") as stdout_f:
        with open(os.path.join(processed_folder, "carball.e.log"), "w", encoding="utf8") as stderr_f:
            return subprocess.run(
                CARBALL_COMMAND.format(carball_path, replay_path, processed_folder),
                stdout=stdout_f,
                stderr=stderr_f,
                env=ENV
            )


def load_parquet(*args, **kwargs):
    return pd.read_parquet(*args, engine="pyarrow", **kwargs)


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


@dataclass
class ParsedReplay:
    metadata: dict
    analyzer: dict
    game_df: pd.DataFrame
    ball_df: pd.DataFrame
    player_dfs: Dict[str, pd.DataFrame]

    @staticmethod
    def load(replay_dir, carball_path=None) -> "ParsedReplay":
        if isinstance(replay_dir, str):
            replay_dir = Path(replay_dir)

        def load_files(rdir):
            with (rdir / "metadata.json").open("r", encoding="utf8") as f:
                metadata = json.load(f)
            with (rdir / "analyzer.json").open("r", encoding="utf8") as f:
                analyzer = json.load(f)
            ball_df = load_parquet(rdir / "__ball.parquet")
            game_df = load_parquet(rdir / "__game.parquet")

            player_dfs = {}
            for player_file in rdir.glob("player_*.parquet"):
                player_id = player_file.name.split("_")[1].split(".")[0]
                player_dfs[player_id] = load_parquet(player_file)

            return ParsedReplay(metadata, analyzer, game_df, ball_df, player_dfs)

        if not replay_dir.is_dir():
            # Assume it's a replay file
            with tempfile.TemporaryDirectory() as temp_dir:
                process_replay(replay_dir, temp_dir, carball_path=carball_path, skip_existing=False)
                replay_dir = Path(temp_dir) / replay_dir.stem
                return load_files(replay_dir)
        else:
            return load_files(replay_dir)

    def to_rlgym(self, interpolate=False, debug=False):
        players = {p["unique_id"]: p for p in self.metadata["players"]}
        transition_engine = RocketSimEngine()
        count_orange = sum(p["is_orange"] for p in players.values())
        base_mutator = FixedTeamSizeMutator(blue_size=len(players) - count_orange, orange_size=count_orange)
        kickoff_mutator = KickoffMutator()

        for gameplay_period in self.analyzer["gameplay_periods"]:
            start_frame = gameplay_period["start_frame"]
            goal_frame = gameplay_period.get("goal_frame")
            end_frame = goal_frame or gameplay_period["end_frame"]

            base_state = transition_engine.create_base_state()

            base_mutator.apply(base_state, {})  # Let base mutator create the cars
            cars = list(base_state.cars.values())

            new_cars = {}
            actions = {}
            for uid, car in zip(self.player_dfs, cars):
                player = players[uid]
                car.hitbox_type = OCTANE  # TODO: Get hitbox from replay
                car.team_num = ORANGE_TEAM if player["is_orange"] else BLUE_TEAM
                new_cars[uid] = car
                actions[uid] = np.zeros((120, 8), dtype=np.float32)
            base_state.cars = new_cars

            kickoff_mutator.apply(base_state, {})
            transition_engine.set_state(base_state, {})
            # Step with no actions to let RocketSim calculate internal values
            state = transition_engine.step(actions, {})

            game_df = self.game_df.loc[start_frame:end_frame]
            ball_df = self.ball_df.loc[start_frame:end_frame].ffill().fillna(0.)
            player_dfs = {}
            for uid, pdf in self.player_dfs.items():
                pdf = pdf.loc[start_frame:end_frame].astype(float)
                interpolate_cols = ([f"{col}_{axis}" for col in ("pos", "vel", "ang_vel") for axis in "xyz"] +
                                    [f"quat_{axis}" for axis in "wxyz"])
                is_repeat = (pdf[interpolate_cols].diff() == 0).all(axis=1)
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
                    pdf.loc[is_repeat, interpolate_cols] = np.nan
                    pdf = pdf.interpolate()  # Note that this assumes equal time steps
                player_dfs[uid] = pdf

            game_config = GameConfig()
            game_config.gravity = 1
            game_config.boost_consumption = 1
            game_config.dodge_deadzone = 0.5

            game_tuples = list(game_df.itertuples())
            player_ids = []
            player_rows = []
            for pid, pdf in player_dfs.items():
                player_ids.append(pid)
                player_rows.append(list(pdf.itertuples()))

            for i, game_row, ball_row, car_rows in zip(range(len(game_tuples)),
                                                       game_tuples,
                                                       ball_df.itertuples(),
                                                       zip(*player_rows)):
                frame = game_row.Index
                state.ball.position[:] = (ball_row.pos_x, ball_row.pos_y, ball_row.pos_z)
                state.ball.linear_velocity[:] = (ball_row.vel_x, ball_row.vel_y, ball_row.vel_z)
                state.ball.angular_velocity[:] = (ball_row.ang_vel_x, ball_row.ang_vel_y, ball_row.ang_vel_z)
                state.ball.quaternion = np.array((ball_row.quat_w, ball_row.quat_x,
                                                  ball_row.quat_y, ball_row.quat_z))

                state.tick_count = game_row.time * TICKS_PER_SECOND
                state.goal_scored = frame == goal_frame
                state.config = game_config

                actions = {}
                for j, uid, player_row in zip(range(len(car_rows)), player_ids, car_rows):
                    car = state.cars[uid]
                    if interpolate or not player_row.is_repeat and (not debug or i % 30 == 0):
                        if debug:
                            pred_pos = car.physics.position.copy()
                            pred_vel = car.physics.linear_velocity.copy()
                            pred_ang_vel = car.physics.angular_velocity.copy()
                            pred_quat = car.physics.quaternion.copy()

                        car.physics.position[:] = (player_row.pos_x, player_row.pos_y, player_row.pos_z)
                        car.physics.linear_velocity[:] = (player_row.vel_x, player_row.vel_y, player_row.vel_z)
                        car.physics.angular_velocity[:] = (
                            player_row.ang_vel_x, player_row.ang_vel_y, player_row.ang_vel_z)
                        car.physics.quaternion = np.array((player_row.quat_w, player_row.quat_x,
                                                           player_row.quat_y, player_row.quat_z))

                        if debug:
                            diff_pos = np.linalg.norm(pred_pos - car.physics.position)
                            diff_vel = np.linalg.norm(pred_vel - car.physics.linear_velocity)
                            diff_ang_vel = np.linalg.norm(pred_ang_vel - car.physics.angular_velocity)
                            diff_quat = np.linalg.norm(pred_quat - car.physics.quaternion)
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
                        if not car.on_ground:
                            asd = 1
                        car.on_ground = True
                        car.has_jumped = False
                        jump = 1
                    elif player_row.dodged:
                        if car.on_ground or car.is_jumping or car.is_holding_jump or not car.can_flip:
                            asd = 1
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
                            pitch /= mx
                            roll /= mx
                        else:
                            # Stall, happens when roll+yaw=0 and abs(roll)+abs(yaw)>deadzone
                            pitch = 0
                            roll = 1
                            yaw = -1
                        jump = 1
                        if debug:
                            dodge_torque = np.array([player_row.dodge_torque_x, player_row.dodge_torque_y, 0])
                            dodge_torque = dodge_torque / np.linalg.norm(dodge_torque)
                            print(f"Player {uid} dodged at frame {frame} ({dodge_torque})")
                    elif player_row.dodge_is_active:
                        # TODO handle flip cancels
                        pitch = yaw = roll = 0
                        if debug:
                            print(f"Player {uid} should be dodging at frame {frame} ({car.flip_torque}) "
                                  f"{car.is_flipping=}")
                    elif player_row.double_jumped:
                        car.on_ground = False
                        car.is_jumping = False
                        car.is_holding_jump = False
                        car.has_flipped = False
                        car.has_double_jumped = False
                        mx = max(abs(pitch), abs(yaw), abs(roll))
                        if mx >= game_config.dodge_deadzone:
                            # Would not have been a double jump, but a dodge. Correct it
                            limiter = 0.98 * game_config.dodge_deadzone / mx
                            pitch = pitch * limiter
                            roll = roll * limiter
                            yaw = yaw * limiter
                        jump = 1

                    boost = player_row.boost_is_active
                    handbrake = player_row.handbrake
                    action = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
                                      dtype=np.float32)
                    actions[uid] = action

                yield state, actions

                if frame == end_frame:
                    break

                ticks = round(game_tuples[i + 1].time * TICKS_PER_SECOND - state.tick_count)
                for uid, action in actions.items():
                    actions[uid] = action.reshape(1, -1).repeat(ticks, axis=0)
                transition_engine.set_state(state, {})
                state = transition_engine.step(actions, {})
