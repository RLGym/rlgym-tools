import glob
import random
import time

import numpy as np
from rlgym.rocket_league.common_values import TICKS_PER_SECOND
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer

from rlgym_tools.replays.convert import replay_to_rlgym, ReplayFrame
from rlgym_tools.replays.parsed_replay import ParsedReplay
from rlgym_tools.reward_functions.flip_reset_reward import FlipResetReward


def main():
    # renderer = RLViserRenderer(tick_rate=30)
    renderer = None
    # random.seed(0)
    # np.random.seed(0)
    paths = glob.glob(r"E:/rokutleg/behavioral-cloning/replays/*/*.replay")
    random.shuffle(paths)
    forfeits = []
    for path in paths:
        print(path)
        # replay = ParsedReplay.load("./test_replays/00029e4d-242d-49ed-971d-1218daa2eefa.replay")
        try:
            replay = ParsedReplay.load(path)
        except FileNotFoundError:
            continue
        flip_reset_reward = FlipResetReward()
        t = 0
        t0 = time.time()
        for replay_frame in replay_to_rlgym(replay, calculate_error=False):
            if t == 0:
                flip_reset_reward.reset(list(replay_frame.state.cars.keys()), replay_frame.state, {})

            rewards = flip_reset_reward.get_rewards(list(replay_frame.state.cars.keys()), replay_frame.state, {}, {},
                                                    {})
            if replay_frame.scoreboard.go_to_kickoff or replay_frame.scoreboard.is_over:
                print(replay_frame.scoreboard)

            # print(state)
            if renderer is not None:
                if replay_frame.scoreboard.kickoff_timer_seconds < 5:
                    t1 = time.time()
                    delta = (replay_frame.state.tick_count - t) / TICKS_PER_SECOND
                    delay = delta - (t1 - t0)
                    if 0 < delay < 1:
                        time.sleep(delay)
                    t0 = time.time()
                renderer.render(replay_frame.state, {})
            t = replay_frame.state.tick_count
            # if t < TICKS_PER_SECOND * 60:
            #     print(next(iter(replay_frame.state.cars.values())).physics.position)
        print("Game over!", replay_frame.scoreboard)
        if not replay_frame.scoreboard.is_over:
            forfeits.append((replay_frame.scoreboard.game_timer_seconds,
                             replay_frame.scoreboard.blue_score - replay_frame.scoreboard.orange_score))
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
