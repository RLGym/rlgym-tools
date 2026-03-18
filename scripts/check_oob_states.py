"""
Check replay mutator states for out-of-bounds positions after stepping RocketSim.

Loads replay states via ReplayMutator, resets an RLGym environment to each state,
steps the simulation, and checks if any cars or the ball end up outside the field.

Usage:
    python scripts/check_oob_states.py <replay_frames.npy> [--resets 1000] [--steps 10] [--margin 200]
"""

import argparse
import sys

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.state_mutators.replay_mutator import ReplayMutator

# Field boundaries
SIDE_WALL_X = 4096
BACK_WALL_Y = 5120
CEILING_Z = 2044
GOAL_X = 893
GOAL_Y = 6000
GOAL_Z = 642.775


def is_oob(pos, margin):
    """Check if a position is out of bounds. Returns a description string or None."""
    x, y, z = pos[0], pos[1], pos[2]
    in_goal_area = abs(x) < GOAL_X + margin and z < GOAL_Z + margin and abs(y) > BACK_WALL_Y

    reasons = []
    if abs(x) > SIDE_WALL_X + margin:
        reasons.append(f"x={x:.0f} (limit ±{SIDE_WALL_X})")
    if abs(y) > BACK_WALL_Y + margin and not in_goal_area:
        reasons.append(f"y={y:.0f} (limit ±{BACK_WALL_Y})")
    if abs(y) > GOAL_Y + margin:
        reasons.append(f"y={y:.0f} (behind goal, limit ±{GOAL_Y})")
    if z < -margin:
        reasons.append(f"z={z:.0f} (below ground)")
    if z > CEILING_Z + margin:
        reasons.append(f"z={z:.0f} (above ceiling {CEILING_Z})")
    return ", ".join(reasons) if reasons else None


def check_state(engine, margin):
    """Check the current engine state for OOB. Returns list of violation strings."""
    violations = []
    state = engine.state

    ball_pos = state.ball.position
    oob = is_oob(ball_pos, margin)
    if oob:
        violations.append(f"BALL OOB — {oob} (pos={np.round(ball_pos, 1)})")

    for agent_id, car in state.cars.items():
        if car.is_demoed:
            continue
        car_pos = car.physics.position
        oob = is_oob(car_pos, margin)
        if oob:
            team = "Blue" if car.is_blue else "Orange"
            violations.append(f"Car {agent_id} ({team}) OOB — {oob} (pos={np.round(car_pos, 1)})")

    return violations


def main():
    parser = argparse.ArgumentParser(description="Check replay states for OOB after stepping RocketSim.")
    parser.add_argument("replay_file", type=str, help="Path to a .npy replay frames file")
    parser.add_argument("--resets", type=int, default=100_000, help="Number of random resets to test (default 1000)")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps per reset (default 10)")
    parser.add_argument("--margin", type=float, default=200, help="OOB tolerance in unreal units (default 200)")
    args = parser.parse_args()

    print(f"Loading {args.replay_file}...")
    replay_mutator = ReplayMutator(args.replay_file)
    num_frames = len(replay_mutator.replay_frames)
    print(f"Loaded {num_frames} frames.")

    engine = RocketSimEngine()
    env = RLGym(
        state_mutator=replay_mutator,
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction()),
        reward_fn=GoalReward(),
        transition_engine=engine,
        termination_cond=GoalCondition(),
    )

    total_violations = 0
    violation_examples = []

    print(f"Running {args.resets} resets with {args.steps} steps each...")
    for reset_i in range(args.resets):
        try:
            obs = env.reset()
        except Exception as e:
            print(f"  Reset {reset_i}: failed — {e}")
            continue

        # Check right after reset
        for v in check_state(engine, args.margin):
            total_violations += 1
            if len(violation_examples) < 20:
                violation_examples.append(f"  Reset {reset_i}, after reset: {v}")

        # Step and check
        for step_i in range(args.steps):
            try:
                agents = env.agents
                actions = {agent: np.array([8]) for agent in agents}
                obs, rewards, terminated, truncated = env.step(actions)
            except Exception as e:
                total_violations += 1
                if len(violation_examples) < 20:
                    violation_examples.append(f"  Reset {reset_i}, step {step_i}: EXCEPTION — {e}")
                break

            for v in check_state(engine, args.margin):
                total_violations += 1
                if len(violation_examples) < 20:
                    violation_examples.append(f"  Reset {reset_i}, step {step_i}: {v}")

            if any(terminated.values()) or any(truncated.values()):
                break

        if (reset_i + 1) % 100 == 0:
            print(f"  {reset_i + 1}/{args.resets} resets done, {total_violations} violations so far...")

    env.close()

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Resets: {args.resets}, Steps per reset: {args.steps}, Margin: {args.margin}")
    if total_violations == 0:
        print("No out-of-bounds positions found!")
    else:
        print(f"Total violations: {total_violations}")
        print(f"\nExamples:")
        for ex in violation_examples:
            print(ex)

    return 1 if total_violations > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

