from typing import Dict, Any

import numpy as np
from rlgym.api import StateMutator, StateType, RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.api import PhysicsObject
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence

from rlgym_tools.rocket_league.misc.serialize import serialize, deserialize
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomPhysicsMutator
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator


def main():
    class RenameAgentMutator(StateMutator):
        def apply(self, state: StateType, shared_info: Dict[str, Any]) -> None:
            state.cars = {i: v for i, v in enumerate(state.cars.values())}

    transition_engine = RocketSimEngine()
    env = RLGym(
        state_mutator=MutatorSequence(
            VariableTeamSizeMutator({(i, j): 1 for i in range(1, 12) for j in range(1, 12)}),
            RandomPhysicsMutator(),
            RenameAgentMutator()
        ),
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction()),
        reward_fn=GoalReward(),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
    )

    obs = env.reset()
    states = []
    serialized_states = []
    while True:
        state = transition_engine.state
        states.append(state)
        s = serialize(state)
        serialized_states.append(s)
        d = deserialize(s)

        replay_frame = ReplayFrame(
            state=state,
            actions={i: np.random.normal(0, 1, size=8) for i in state.cars.keys()},
            update_age={i: 0.0 for i in state.cars.keys()},
            scoreboard=ScoreboardInfo(
                game_timer_seconds=np.random.uniform(0, 300),
                kickoff_timer_seconds=np.random.uniform(0, 5),
                blue_score=np.random.randint(0, 10),
                orange_score=np.random.randint(0, 10),
                go_to_kickoff=np.random.choice([True, False]),
                is_over=np.random.choice([True, False]),
            ),
            episode_seconds_remaining=np.random.uniform(0, 300),
            next_scoring_team=np.random.choice([0, 1, None]),
            winning_team=np.random.choice([0, 1, None]),
        )
        s = serialize(replay_frame)
        d: ReplayFrame = deserialize(s)

        assert np.isclose(d.scoreboard.game_timer_seconds, replay_frame.scoreboard.game_timer_seconds)
        assert np.isclose(d.scoreboard.kickoff_timer_seconds, replay_frame.scoreboard.kickoff_timer_seconds)
        assert d.scoreboard.blue_score == replay_frame.scoreboard.blue_score
        assert d.scoreboard.orange_score == replay_frame.scoreboard.orange_score
        assert d.scoreboard.go_to_kickoff == replay_frame.scoreboard.go_to_kickoff
        assert d.scoreboard.is_over == replay_frame.scoreboard.is_over
        assert np.isclose(d.episode_seconds_remaining, replay_frame.episode_seconds_remaining)
        assert d.next_scoring_team == replay_frame.next_scoring_team
        assert d.winning_team == replay_frame.winning_team

        assert all(np.allclose(d.actions[i], replay_frame.actions[i]) for i in replay_frame.actions)
        assert all(np.isclose(d.update_age[i], replay_frame.update_age[i]) for i in replay_frame.update_age)

        dstate = d.state
        assert dstate.tick_count == state.tick_count
        assert dstate.goal_scored == state.goal_scored
        assert dstate.config.gravity == state.config.gravity
        assert dstate.config.boost_consumption == state.config.boost_consumption
        assert dstate.config.dodge_deadzone == state.config.dodge_deadzone
        assert np.allclose(dstate.ball.position, state.ball.position)
        assert np.allclose(dstate.ball.linear_velocity, state.ball.linear_velocity)
        assert np.allclose(dstate.ball.angular_velocity, state.ball.angular_velocity)
        assert np.allclose(dstate.ball.rotation_mtx, state.ball.rotation_mtx)
        assert np.allclose(dstate.boost_pad_timers, state.boost_pad_timers)
        assert len(dstate.cars) == len(state.cars)
        car_check = {}
        for i, v in state.cars.items():
            for attr in v.__slots__:
                if attr.startswith("_"):
                    continue
                val = getattr(v, attr)
                other_val = getattr(dstate.cars[i], attr)
                if isinstance(val, np.ndarray):
                    car_check[attr] = np.allclose(getattr(dstate.cars[i], attr), val)
                elif isinstance(val, PhysicsObject):
                    car_check[attr] = np.allclose(val.position, other_val.position) and \
                                      np.allclose(val.linear_velocity, other_val.linear_velocity) and \
                                      np.allclose(val.angular_velocity, other_val.angular_velocity) and \
                                      np.allclose(val.rotation_mtx, other_val.rotation_mtx)
                elif isinstance(val, (int, float, bool)) and isinstance(other_val, (int, float, bool)):
                    car_check[attr] = np.isclose(val, other_val)
                else:
                    car_check[attr] = val == other_val

        assert all(car_check.values()), "Car check failed for" + str({k: v for k, v in car_check.items() if not v})

        actions = np.random.choice(90, size=(len(env.agents),))
        actions = {agent: action.reshape(-1, 1) for agent, action in zip(env.agents, actions)}
        obs, reward, terminated, truncated = env.step(actions)
        print("All checks passed")
        if any(terminated.values()):
            env.reset()


if __name__ == '__main__':
    main()
