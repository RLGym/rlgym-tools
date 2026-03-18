"""
Visualize replay mutator states using an RLGym environment with RocketSimVis.

This creates a proper RLGym environment with:
- ReplayMutator as the state mutator (loads states from a .npy file)
- RocketSimEngine as the transition engine
- RocketSimVisRenderer as the renderer

Each time you press Enter, the environment resets to a new random replay state
and renders it via RocketSimVis.

Usage:
    python scripts/visualize_replay_states.py <replay_frames.npy>

Controls (in the terminal):
    Enter         : Reset to a new random replay state
    s             : Step the environment with no-op actions
    s <n>         : Step n times
    i             : Print info about the current state
    q             : Quit

Requires RocketSimVis to be running for 3D visualization:
    https://github.com/ZealanL/RocketSimVis
"""

import argparse
import time

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
from rlgym_tools.rocket_league.state_mutators.replay_mutator import ReplayMutator


def print_state_info(engine):
    """Print detailed info about the current state."""
    state = engine.state
    print(f"\n--- State Info ---")
    print(f"  Tick: {state.tick_count}")
    print(f"  Ball: pos={np.round(state.ball.position, 1)}, "
          f"vel={np.round(state.ball.linear_velocity, 1)}")
    for agent_id, car in state.cars.items():
        team = "Blue" if car.is_blue else "Orange"
        demo = " [DEMO]" if car.is_demoed else ""
        print(f"  Car {agent_id} ({team}){demo}: "
              f"pos={np.round(car.physics.position, 1)}, "
              f"boost={car.boost_amount:.0%}")


def main():
    parser = argparse.ArgumentParser(description="Visualize replay states using an RLGym environment.")
    parser.add_argument("replay_file", type=str,
                        help="Path to a .npy replay frames file")
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1",
                        help="RocketSimVis UDP IP address")
    parser.add_argument("--udp-port", type=int, default=9273,
                        help="RocketSimVis UDP port")
    args = parser.parse_args()

    print(f"Loading {args.replay_file}...")
    replay_mutator = ReplayMutator(args.replay_file)
    print(f"Loaded {len(replay_mutator.replay_frames)} frames.")

    renderer = RocketSimVisRenderer(udp_ip=args.udp_ip, udp_port=args.udp_port)
    engine = RocketSimEngine()

    env = RLGym(
        state_mutator=replay_mutator,
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction()),
        reward_fn=GoalReward(),
        transition_engine=engine,
        termination_cond=GoalCondition(),
        renderer=renderer,
    )

    print("\nControls:")
    print("  Enter       : Reset to new random replay state")
    print("  s [n]       : Step n times (default 1) with no-op actions")
    print("  p [n] [fps] : Play n steps at fps (default 300 steps, 15 fps)")
    print("  i           : Print state info")
    print("  q           : Quit")
    print()

    obs = env.reset()
    env.render()
    print(f"Reset. {len(obs)} agents.")

    try:
        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                break

            if cmd == 'q':
                print("Quitting.")
                break
            elif cmd == '':
                obs = env.reset()
                env.render()
                print(f"Reset. {len(obs)} agents.")
            elif cmd.startswith('s'):
                parts = cmd.split()
                n = 1
                if len(parts) > 1:
                    try:
                        n = int(parts[1])
                    except ValueError:
                        pass
                # No-op action (index 8 in lookup table = no inputs)
                for step_i in range(n):
                    agents = env.agents
                    actions = {agent: np.array([8]) for agent in agents}
                    obs, rewards, terminated, truncated = env.step(actions)
                    env.render()
                    if any(terminated.values()) or any(truncated.values()):
                        print(f"  Episode ended at step {step_i + 1}. Resetting...")
                        obs = env.reset()
                        env.render()
                        break
                print(f"Stepped {n} time(s).")
            elif cmd.startswith('p'):
                parts = cmd.split()
                n = 30
                fps = 15.0
                if len(parts) > 1:
                    try:
                        n = int(parts[1])
                    except ValueError:
                        pass
                if len(parts) > 2:
                    try:
                        fps = float(parts[2])
                    except ValueError:
                        pass
                print(f"Playing {n} steps at {fps:.0f} FPS...")
                for step_i in range(n):
                    agents = env.agents
                    actions = {agent: np.array([8]) for agent in agents}
                    obs, rewards, terminated, truncated = env.step(actions)
                    env.render()
                    if any(terminated.values()) or any(truncated.values()):
                        print(f"  Episode ended at step {step_i + 1}. Resetting...")
                        obs = env.reset()
                        env.render()
                    time.sleep(1.0 / fps)
                print(f"Done playing.")
            elif cmd == 'i':
                print_state_info(engine)
            else:
                print(f"Unknown command: {cmd!r}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
        print("Done.")


if __name__ == '__main__':
    main()

