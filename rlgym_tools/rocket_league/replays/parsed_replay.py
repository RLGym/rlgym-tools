import json
import os
import platform
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import stat

import pandas as pd

ENV = os.environ.copy()
ENV["NO_COLOR"] = "1"
ENV["RUST_BACKTRACE"] = "full"

# Use carball executable in the same directory as this script by default
if platform.system() == "Windows":
    _default_executable = os.path.join(os.path.dirname(__file__), "carball.exe")
else:
    _default_executable = os.path.join(os.path.dirname(__file__), "carball")
    # Check if we have the needed permissions, and try to update if we don't
    _st = os.stat(_default_executable)
    if not _st.st_mode & stat.S_IXUSR:
        try:
            os.chmod(_default_executable, _st.st_mode | stat.S_IXUSR)
        except PermissionError:
            warnings.warn(
                f"Could not set executable permissions for {os.path.basename(_default_executable)}. "
                "Please ensure it is executable by the current user, or specify the path to the carball executable "
                "using the `carball_path` argument.",
                UserWarning
            )


def process_replay(replay_path, output_folder, carball_path=None, skip_existing=True):
    if carball_path is None:
        carball_path = _default_executable
    folder, fn = os.path.split(replay_path)
    replay_name = fn.replace(".replay", "")
    processed_folder = os.path.join(output_folder, replay_name)
    if os.path.isdir(processed_folder) and len(os.listdir(processed_folder)) > 0:
        if skip_existing:
            return
        else:
            os.rmdir(processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    carball_command = [
        carball_path, "parquet",
        "-i", replay_path,
        "-o", processed_folder,
    ]

    with open(os.path.join(processed_folder, "carball.o.log"), "w", encoding="utf8") as stdout_f:
        with open(os.path.join(processed_folder, "carball.e.log"), "w", encoding="utf8") as stderr_f:
            try:
                return subprocess.run(
                    carball_command,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=ENV
                )
            except PermissionError:
                os.chmod(carball_path, 0o755)
                return subprocess.run(
                    carball_command,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=ENV
                )


def load_parquet(*args, **kwargs):
    return pd.read_parquet(*args, engine="pyarrow", **kwargs)


@dataclass
class ParsedReplay:
    metadata: dict
    analyzer: dict
    game_df: pd.DataFrame
    ball_df: pd.DataFrame
    player_dfs: Dict[str, pd.DataFrame]

    @staticmethod
    def load(replay_dir, carball_path=None) -> "ParsedReplay":
        if not os.path.exists(replay_dir):
            raise FileNotFoundError(f"Replay directory {replay_dir} does not exist")

        if isinstance(replay_dir, str):
            replay_dir = Path(replay_dir)

        def load_files(rdir):
            try:
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
            except FileNotFoundError as e:
                try:
                    with (rdir / "carball.e.log").open("r", encoding="utf8") as f:
                        raise ValueError(f"Error processing replay: {f.read()}")
                except FileNotFoundError:
                    raise e

        if not replay_dir.is_dir():
            # Assume it's a replay file
            with tempfile.TemporaryDirectory() as temp_dir:
                process_replay(replay_dir, temp_dir, carball_path=carball_path, skip_existing=False)
                replay_dir = Path(temp_dir) / replay_dir.stem
                return load_files(replay_dir)
        else:
            return load_files(replay_dir)
