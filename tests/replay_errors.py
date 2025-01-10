import functools
import glob
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from rlgym.rocket_league.api import Car

from rlgym_tools.rocket_league.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym_tools.rocket_league.misc.action import Action
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
    paths = glob.glob(r"./test_replays/36d77ff2-49b8-42e8-9012-c238f0295e31.replay")
    # paths = glob.glob(r"./test_replays/bb772d09-5d6f-4174-8d57-b0f6853e1638.replay")
    assert paths, "No replays found"
    random.shuffle(paths)
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

    for path in paths:
        print(path)

        replay = ParsedReplay.load(path)
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
                total_pos_error = {}
                total_quat_error = {}
                total_vel_error = {}
                total_ang_vel_error = {}

                for replay_frame, errors in replay_to_rlgym(replay, calculate_error=True,
                                                            modify_action_fn=modify_action):
                    replay_frame: ReplayFrame
                    scoreboard = replay_frame.scoreboard
                    for uid, car in replay_frame.state.cars.items():
                        if uid in errors:
                            player_errors = errors[uid]

                            total_pos_error.setdefault(uid, []).append(player_errors["pos"])
                            total_quat_error.setdefault(uid, []).append(player_errors["quat"])
                            total_vel_error.setdefault(uid, []).append(player_errors["vel"])
                            total_ang_vel_error.setdefault(uid, []).append(player_errors["ang_vel"])

                        action = replay_frame.actions[uid]
                        # print(scoreboard)
                        # print(f"{uid}, {car.on_ground=}, {car.boost_amount=:.2f}, {car.can_flip=}, {car.is_flipping=}")
                        # print("replay action:\n\t" + str(Action.from_numpy(action)))
                        #
                        # t0 = time.perf_counter()
                        # probs = method(car, action, vortex_lut)
                        # t1 = time.perf_counter()
                        # idxs = np.where(probs > 0)[0]
                        # s = f"{method.__name__}: ({(t1 - t0) * 1000:.3f}ms)\n"
                        # if len(idxs) > 1:
                        #     weighted_average = np.average(vortex_lut, axis=0, weights=probs)
                        #     s += f"WA: {Action.from_numpy(weighted_average)}\n"
                        # for idx in sorted(idxs, key=lambda k: -probs[k]):
                        #     a = vortex_lut[idx]
                        #     a = Action.from_numpy(a)
                        #     s += f"\t{a} ({probs[idx]:.0%})\n"
                        #     print(s[:-1])
                        # print()
                # fig, axs = plt.subplots(nrows=4)
                i = 0
                print(f"Method: {method.__name__ if method is not None else 'None'}, "
                      f"Lookup table: {lut_name}")
                for name, error in [("Position", total_pos_error), ("Quaternion", total_quat_error),
                                    ("Velocity", total_vel_error), ("Angular velocity", total_ang_vel_error)]:
                    for uid, errors in error.items():
                        error[uid] = np.array(errors)
                        # axs[i].plot(error[uid], label=uid)
                    all_errors = np.concatenate(list(error.values()))

                    print(f"{name} error: \n"
                          f"\tMean: {np.mean(all_errors):.6g}\n"
                          f"\tStd: {np.std(all_errors):.6g}\n"
                          f"\tMedian: {np.median(all_errors):.6g}\n"
                          f"\tMax: {np.max(all_errors):.6g}")
                    # print(f"{name} error:")
                    # print("\n".join([str(np.mean(all_errors)), str(np.std(all_errors)),
                    #                  str(np.median(all_errors)), str(np.max(all_errors))]))
                    # axs[i].hist(all_errors, bins=100)
                    # axs[i].set_yscale("log")
                    # axs[i].set_title(name)
                    i += 1
                print()
                # fig.legend()
                # plt.show()


if __name__ == '__main__':
    main()
