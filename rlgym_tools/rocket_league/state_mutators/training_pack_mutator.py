import glob
import gzip
import json
import os
import random
import warnings
from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState, PhysicsObject

from rlgym_tools.rocket_league.replays.training_packs import training_pack_to_json
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo


class TrainingPackMutator(StateMutator[GameState]):
    def __init__(self, training_packs: str | dict | list | None = None):
        self.training_packs = self._load(training_packs)

        code_to_idx = {}
        tag_to_idx = {}
        type_to_idx = {}
        diff_to_idx = {}
        for idx, pack in enumerate(self.training_packs):
            meta = pack.get("metadata", {})
            code = meta.get("code")
            tags = meta.get("tags", [])
            training_type = meta.get("type")
            diff = meta.get("difficulty")
            if code:
                code_to_idx[code] = idx
            if tags:
                for tag in tags:
                    if tag not in tag_to_idx:
                        tag_to_idx[tag] = []
                    tag_to_idx[tag].append(idx)
            if training_type:
                if training_type not in type_to_idx:
                    type_to_idx[training_type] = []
                type_to_idx[training_type].append(idx)
            if diff:
                if diff not in diff_to_idx:
                    diff_to_idx[diff] = []
                diff_to_idx[diff].append(idx)
        self._code_to_idx = code_to_idx
        self._tag_to_idx = tag_to_idx
        self._type_to_idx = type_to_idx
        self._diff_to_idx = diff_to_idx

    @staticmethod
    def _load(training_packs: str | dict | list | None) -> list[dict]:
        if training_packs is None:
            return TrainingPackMutator._load("./training_packs.jsonl.gz")
        if isinstance(training_packs, list):
            return training_packs
        elif isinstance(training_packs, dict):
            return [training_packs]
        elif training_packs.startswith("["):
            return json.loads(training_packs)
        elif training_packs.startswith("{"):
            return [json.loads(training_packs)]
        else:
            path = training_packs
            if path.endswith(".gz"):
                f = gzip.open(path, "rt")
                path = path[:-3]
            else:
                f = open(path, "r")
            with f:
                if path.endswith(".jsonl"):
                    return [json.loads(line) for line in f]
                else:
                    return json.load(f)

    @staticmethod
    def _write(training_packs: list[dict], out_path=None) -> None | list[dict]:
        if out_path is None:
            return training_packs
        path = out_path
        if path.endswith(".gz"):
            f = gzip.open(path, "wt")
            path = path[:-3]
        else:
            f = open(path, "w")
        with f:
            if path.endswith(".jsonl"):
                for pack in training_packs:
                    f.write(json.dumps(pack) + "\n")
            else:
                json.dump(training_packs, f, indent=2)
        return None

    def _determine_idx(self, shared_info):
        requested = shared_info.get("training_pack_request")
        pack_idx = None
        shot_idx = None
        if requested:
            if "code" in requested:
                pack_idx = self._code_to_idx.get(requested["code"])
            elif "tag" in requested:
                pack_idx = self._tag_to_idx.get(requested["tag"])
            elif "type" in requested:
                pack_idx = self._type_to_idx.get(requested["type"])
            elif "difficulty" in requested:
                pack_idx = self._diff_to_idx.get(requested["difficulty"])
            elif "title" in requested:
                title = requested["title"].lower()
                pack_idx = [i for i, p in enumerate(self.training_packs)
                            if title in p.get("metadata", {}).get("title", "").lower()]
            if pack_idx is None:
                warnings.warn(f"Requested training pack not found: {requested}, falling back to random")
            if isinstance(pack_idx, list):
                pack_idx = random.choice(pack_idx)
            if "shot" in requested:
                shot_idx = requested["shot"]
        if pack_idx is None:
            pack_idx = random.randrange(len(self.training_packs))

        return pack_idx, shot_idx

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        pack_idx, shot_idx = self._determine_idx(shared_info)
        pack = self.training_packs[pack_idx]

        shots = pack.get("shots", [])

        shot = random.choice(shots) if shot_idx is None else shots[shot_idx]

        state.config.boost_consumption = 0.0  # Training packs have infinite boost

        state.ball.position = np.array(shot["ball"]["position"])
        state.ball.linear_velocity = np.array(shot["ball"]["linear_velocity"])
        state.ball.euler_angles = np.array(shot["ball"]["euler_angles"])
        state.ball.angular_velocity = np.zeros(3)

        car = next(car for car in state.cars.values() if car.is_blue)  # Assume blue team, single car
        car.boost_amount = 100
        car_physics = PhysicsObject()
        car_physics.position = np.array(shot["car"]["position"])
        car_physics.linear_velocity = np.array(shot["car"]["linear_velocity"])
        car_physics.euler_angles = np.array(shot["car"]["euler_angles"])
        car_physics.angular_velocity = np.zeros(3)
        car.physics = car_physics
        state.cars = {"blue-0": car}

        if "scoreboard" in shared_info:
            # Simulate a scenario where it has to score before time runs out, or lose the game
            sb: ScoreboardInfo = shared_info["scoreboard"]
            sb.blue_score = 0
            sb.orange_score = 1
            sb.kickoff_timer_seconds = 0
            sb.game_timer_seconds = shot["time_limit"]

        shared_info["training_pack"] = {"pack": pack, "shot_idx": shot_idx}

    @staticmethod
    def make_file(in_path=None, out_path=None):
        if in_path is None:
            # Educated guess. Favorite training packs are stored here
            in_path = r"C:\Users\*\Documents\My Games\Rocket League\TAGame\Training\*\Favorities"
        if os.path.isdir(in_path):
            query = os.path.join(in_path, "**", "*.Tem")
            files = glob.glob(query, recursive=True)
        else:
            files = [in_path]
        if not files:
            raise ValueError(f"No training pack files found at: {in_path}")
        seen_codes = set()
        all_packs = []
        for fpath in files:
            j = training_pack_to_json(fpath)
            code = j.get("metadata", {}).get("code")
            if code in seen_codes:
                continue
            seen_codes.add(code)
            all_packs.append(j)

        return TrainingPackMutator._write(all_packs, out_path)
