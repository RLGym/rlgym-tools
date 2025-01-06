import glob
import random
import time

import matplotlib.pyplot as plt
from rlgym.rocket_league.common_values import TICKS_PER_SECOND

from rlgym_tools.rocket_league.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.reward_functions.flip_reset_reward import FlipResetReward


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
    paths = glob.glob(r"./test_replays/bb772d09-5d6f-4174-8d57-b0f6853e1638.replay")
    random.shuffle(paths)
    forfeits = {}
    # lookup_table = AdvancedLookupTableAction.make_lookup_table(torque_bins=5,
    #                                                            flip_bins=16, include_stalls=True)
    lookup_table = AdvancedLookupTableAction.make_lookup_table()
    win_probs = []
    for path in paths:
        print(path)
        # replay = ParsedReplay.load("./test_replays/00029e4d-242d-49ed-971d-1218daa2eefa.replay")
        # try:
        replay = ParsedReplay.load(path)
        # except FileNotFoundError:
        #     continue
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

            if any(reward > 0 for reward in rewards.values()):
                print(f"Flip reset! {rewards}")

            parsed_actions = {}
            for agent_id, car in replay_frame.state.cars.items():
                action = replay_frame.actions[agent_id]

                # For debugging:
                # probs0 = get_simple_action_options(car, action, lookup_table)
                # probs1 = get_weighted_action_options(car, action, lookup_table)
                # probs2 = get_best_action_options(car, action, lookup_table)

            # print(state)
            if renderer is not None:
                if replay_frame.scoreboard.kickoff_timer_seconds < 5:
                    t1 = time.time()
                    delta = (replay_frame.state.tick_count - t) / TICKS_PER_SECOND
                    delay = delta - (t1 - t0)
                    if 0 < delay < 1:
                        time.sleep(delay)
                    t0 = time.time()
                renderer.render(replay_frame.state, {"controls": replay_frame.actions})
            t = replay_frame.state.tick_count
            # if t < TICKS_PER_SECOND * 60:
            #     print(next(iter(replay_frame.state.cars.values())).physics.position)
        print("Game over!", replay_frame.scoreboard)
        if not replay_frame.scoreboard.is_over:
            forfeits.setdefault(len(replay_frame.state.cars), []).append(
                (replay_frame.scoreboard.game_timer_seconds,
                 replay_frame.scoreboard.blue_score - replay_frame.scoreboard.orange_score)
            )
            fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 10))
            for i, (k, v) in enumerate(sorted(forfeits.items())):
                x, y = zip(*v)
                y = [abs(yi) for yi in y]
                axs[i].scatter(x, y)
                axs[i].grid()
                axs[i].set_title(f"{i + 1}v{i + 1}")
            fig.tight_layout()
            fig.savefig("forfeit.png")
            plt.close(fig)
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
