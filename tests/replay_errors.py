import glob
import os.path
import random
import warnings

import numpy as np
from rlgym.rocket_league.api import Car

from rlgym_tools.rocket_league.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.pick_action import get_weighted_action_options, get_best_action_options
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame


def main():
    # renderer = RLViserRenderer(tick_rate=30)
    # renderer = RocketSimVisRenderer()
    renderer = None
    # random.seed(0)
    # np.random.seed(0)
    # paths = glob.glob(r"E:/rokutleg/behavioral-cloning/replays/*/*.replay")
    # paths = glob.glob(r"./test_replays/801f43e3-3f0d-4b0a-a954-631ca35321db.replay")
    # paths = glob.glob(r"./test_replays/0328fc07-13fb-4cb6-9a86-7d608196ddbd.replay")
    # paths = glob.glob(r"./test_replays/2e786f1b-d3a1-4e51-9e7a-b45e80b10abb.replay")
    # paths = glob.glob(r"./test_replays/1168d253-8deb-4454-afba-cb6b09d7b3fc.replay")
    # paths = glob.glob(r"./test_replays/6414a7ae-0e30-4e0d-8fe6-6769ec540852.replay")
    # paths = glob.glob(r"./test_replays/36d77ff2-49b8-42e8-9012-c238f0295e31.replay")
    # paths = glob.glob(r"./test_replays/bb772d09-5d6f-4174-8d57-b0f6853e1638.replay")
    paths = glob.glob(r"./test_replays/suite/*.replay")
    # paths = glob.glob(r"E:/rokutleg/replays/assorted/**/*.replay", recursive=True)
    assert paths, "No replays found"
    # random.shuffle(paths)
    paths = sorted(paths)
    forfeits = {}
    # vortex_lut = AdvancedLookupTableAction.make_lookup_table(torque_bins=5,
    #                                                            flip_bins=16, include_stalls=True)
    base_lut = AdvancedLookupTableAction.make_lookup_table()
    vortex_lut = AdvancedLookupTableAction.make_lookup_table(
        throttle_bins=(-1, 0, 0.5, 1),  # -0.5 is not very useful
        steer_bins=5,
        torque_subdivisions=3,  # Default meshgrid plus inbetween values (4 per cube face)
        flip_bins=12,  # Gives 60 degree diagonals, closer to optimal speed-flip
        include_stalls=True,
    )
    flip_lut = AdvancedLookupTableAction.make_lookup_table(
        flip_bins=64,
    )
    torque_lut = AdvancedLookupTableAction.make_lookup_table(
        torque_subdivisions=5,
    )

    methods = [
        None,
        get_best_action_options,
        get_weighted_action_options,
    ]
    tables = [
        base_lut,
        vortex_lut,
        flip_lut,
        torque_lut,
    ]
    table_names = [
        "Base",
        "Vortex",
        "Flip",
        "Torque",
    ]

    random.seed(0)
    np.random.seed(0)

    cached_replays = {}  # So we don't have to reload the same replay multiple times

    for lut_name, lut in zip(table_names, tables):
        for method in methods:
            if method is None:
                if not (lut is tables[0]):
                    continue  # Only do None once (lookup table is ignored)
                modify_action = None
            else:
                def modify_action(car: Car, action: np.ndarray):
                    probs = method(car, action, lut)
                    idx = np.random.choice(len(probs), p=probs)
                    return lut[idx]
            all_errors = {}
            for path in paths:
                # print(path)

                if path in cached_replays:
                    replay = cached_replays[path]
                else:
                    replay = ParsedReplay.load(path)
                    if len(paths) < 10:
                        cached_replays[path] = replay

                update_rate = 1 / replay.game_df["delta"].median()
                if not 25 < update_rate < 35:
                    replay_id = os.path.basename(path).split(".")[0]
                    warnings.warn(f"Replay {replay_id} is not close to 30Hz ({update_rate:.2f}Hz)")

                total_errors = {}

                for replay_frame, errors in replay_to_rlgym(replay, calculate_error=True,
                                                            modify_action_fn=modify_action):
                    replay_frame: ReplayFrame
                    scoreboard = replay_frame.scoreboard
                    for uid, car in replay_frame.state.cars.items():
                        if uid in errors:
                            player_errors = errors[uid]

                            terr = total_errors.setdefault(uid, {})
                            for key in player_errors.keys():
                                value = player_errors.get(key, np.nan)
                                terr.setdefault(key, []).append(value)
                            for key in set(terr.keys()) - set(player_errors.keys()):
                                terr[key].append(np.nan)

                        action = replay_frame.actions[uid]

                for uid, error in total_errors.items():
                    for name, errors in error.items():
                        all_errors.setdefault(name, []).extend(errors)

            print(f"Method: {method.__name__ if method is not None else 'None'}, "
                  f"Lookup table: {lut_name}")
            for name, errors in all_errors.items():
                print(f"{name} error: \n"
                      f"\tMean: {np.nanmean(errors):.6g}\n"
                      f"\tStd: {np.nanstd(errors):.6g}\n"
                      f"\tMedian: {np.nanmedian(errors):.6g}\n"
                      f"\tMax: {np.nanmax(errors):.6g}")
            print()


if __name__ == '__main__':
    main()
