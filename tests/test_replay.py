import time

from rlgym.rocket_league.common_values import TICKS_PER_SECOND
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer

from rlgym_tools.replays.convert import replay_to_rlgym
from rlgym_tools.replays.parsed_replay import ParsedReplay


def main():
    replay = ParsedReplay.load("./test_replays/00029e4d-242d-49ed-971d-1218daa2eefa.replay")
    renderer = RLViserRenderer(tick_rate=30)
    t = 0
    for state, actions, *_ in (replay_to_rlgym(replay, debug=True)):
        # print(state)
        time.sleep((state.tick_count - t) / TICKS_PER_SECOND)
        renderer.render(state, {})
        t = state.tick_count
    renderer.close()


if __name__ == "__main__":
    main()
