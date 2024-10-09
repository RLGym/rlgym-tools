from typing import Literal, Dict, Any

import numpy as np
from rlgym.api import RLGym, StateMutator, StateType
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.api import GameState, Car, PhysicsObject, GameConfig
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym.rocket_league.done_conditions import GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator, MutatorSequence

from rlgym_tools.shared_info_providers.scoreboard_provider import ScoreboardInfo


# Utilities to serialize and deserialize RLGym Rocket League objects into numpy arrays
# This can be useful for saving and loading states, sending over network, or batch operations.


def serialize(obj):
    if isinstance(obj, GameState):
        return serialize_game_state(obj)
    elif isinstance(obj, GameConfig):
        return serialize_config(obj)
    elif isinstance(obj, Car):
        return serialize_car(obj)
    elif isinstance(obj, PhysicsObject):
        return serialize_physics_object(obj)
    elif isinstance(obj, ScoreboardInfo):
        return serialize_scoreboard(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def deserialize(obj):
    if isinstance(obj, np.ndarray):
        if obj.shape == (18,):
            return deserialize_physics_object(obj)
        elif obj.shape == (42,):
            return deserialize_car(obj)
        elif obj.shape == (3,):
            return deserialize_config(obj)
        elif obj.shape == (6,):
            return deserialize_scoreboard(obj)
        else:
            return deserialize_game_state(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def serialize_game_state(game_state: GameState) -> np.ndarray:
    return np.concatenate([
        np.array([game_state.tick_count,
                  int(game_state.goal_scored)]),
        serialize_config(game_state.config),
        serialize_physics_object(game_state.ball),
        game_state.boost_pad_timers,
        *[np.array([agent_id, *serialize_car(car)])
          for agent_id, car in game_state.cars.items()]
    ])


# Indices for serialized GameState:
GS_TICK_COUNT = 0
GS_GOAL_SCORED = 1
GS_CONFIG = slice(2, 5)
GS_BALL = slice(5, 23)
GS_BOOST_PAD_TIMERS = slice(23, 23 + len(BOOST_LOCATIONS))
GS_CARS = slice(23 + len(BOOST_LOCATIONS), None)


def deserialize_game_state(data: np.ndarray) -> GameState:
    assert (len(data) - GS_CARS.start) % 43 == 0, f"Invalid data length: {len(data)}"
    game_state = GameState()
    game_state.tick_count = int(data[GS_TICK_COUNT])
    game_state.goal_scored = bool(data[GS_GOAL_SCORED])

    config = deserialize_config(data[GS_CONFIG])
    game_state.config = config

    ball = deserialize_physics_object(data[GS_BALL])
    game_state.ball = ball

    game_state.boost_pad_timers = data[GS_BOOST_PAD_TIMERS]

    rest = data[GS_CARS]
    cars = {}
    for i in range(0, len(rest), 43):
        agent_id = int(rest[i])
        car = deserialize_car(rest[i + 1:i + 43])
        cars[agent_id] = car
    game_state.cars = cars

    return game_state


def serialize_config(config: GameConfig) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]:
    return np.array([
        config.gravity,
        config.boost_consumption,
        config.dodge_deadzone,
    ], dtype=np.float32)


# Indices for serialized GameConfig:
CONFIG_GRAVITY = 0
CONFIG_BOOST_CONSUMPTION = 1
CONFIG_DODGE_DEADZONE = 2


def deserialize_config(data: np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]) -> GameConfig:
    assert data.shape == (3,)
    config = GameConfig()
    config.gravity = float(data[CONFIG_GRAVITY])
    config.boost_consumption = float(data[CONFIG_BOOST_CONSUMPTION])
    config.dodge_deadzone = float(data[CONFIG_DODGE_DEADZONE])

    return config


def serialize_car(car: Car) -> np.ndarray[tuple[Literal[42]], np.dtype[np.float32]]:
    bump_victim = car.bump_victim_id
    if bump_victim is None:
        bump_victim = -1
    else:
        bump_victim = int(bump_victim)
        assert bump_victim != -1, "Invalid bump victim ID (-1 is reserved for None)"
    return np.concatenate([
        np.array([car.team_num,
                  car.hitbox_type,
                  car.ball_touches,
                  bump_victim,

                  car.demo_respawn_timer,
                  car.on_ground,
                  car.supersonic_time,
                  car.boost_amount,
                  car.boost_active_time,
                  car.handbrake,

                  car.has_jumped,
                  car.is_holding_jump,
                  car.is_jumping,
                  car.jump_time,

                  car.has_flipped,
                  car.has_double_jumped,
                  car.air_time_since_jump,
                  car.flip_time]),
        car.flip_torque,
        np.array([car.is_autoflipping,
                  car.autoflip_timer,
                  car.autoflip_direction]),
        serialize_physics_object(car.physics),
    ], dtype=np.float32)


# Indices for serialized Car:
CAR_TEAM_NUM = 0
CAR_HITBOX_TYPE = 1
CAR_BALL_TOUCHES = 2
CAR_BUMP_VICTIM_ID = 3

CAR_DEMO_RESPAWN_TIMER = 4
CAR_ON_GROUND = 5
CAR_SUPERSONIC_TIME = 6
CAR_BOOST_AMOUNT = 7
CAR_BOOST_ACTIVE_TIME = 8
CAR_HANDBRAKE = 9

CAR_HAS_JUMPED = 10
CAR_IS_HOLDING_JUMP = 11
CAR_IS_JUMPING = 12
CAR_JUMP_TIME = 13

CAR_HAS_FLIPPED = 14
CAR_HAS_DOUBLE_JUMPED = 15
CAR_AIR_TIME_SINCE_JUMP = 16
CAR_FLIP_TIME = 17
CAR_FLIP_TORQUE = slice(18, 21)

CAR_IS_AUTOFLIPPING = 21
CAR_AUTOFLIP_TIMER = 22
CAR_AUTOFLIP_DIRECTION = 23

CAR_PHYSICS = slice(24, 42)


def deserialize_car(data: np.ndarray[tuple[Literal[42]], np.dtype[np.float32]]) -> Car:
    assert data.shape == (42,)
    car = Car()
    car.team_num = int(data[CAR_TEAM_NUM])
    car.hitbox_type = int(data[CAR_HITBOX_TYPE])
    car.ball_touches = int(data[CAR_BALL_TOUCHES])
    car.bump_victim_id = int(data[CAR_BUMP_VICTIM_ID]) if data[CAR_BUMP_VICTIM_ID] != -1 else None

    car.demo_respawn_timer = float(data[CAR_DEMO_RESPAWN_TIMER])
    car.on_ground = bool(data[CAR_ON_GROUND])
    car.supersonic_time = float(data[CAR_SUPERSONIC_TIME])
    car.boost_amount = float(data[CAR_BOOST_AMOUNT])
    car.boost_active_time = float(data[CAR_BOOST_ACTIVE_TIME])
    car.handbrake = float(data[CAR_HANDBRAKE])

    car.has_jumped = bool(data[CAR_HAS_JUMPED])
    car.is_holding_jump = bool(data[CAR_IS_HOLDING_JUMP])
    car.is_jumping = bool(data[CAR_IS_JUMPING])
    car.jump_time = float(data[CAR_JUMP_TIME])

    car.has_flipped = bool(data[CAR_HAS_FLIPPED])
    car.has_double_jumped = bool(data[CAR_HAS_DOUBLE_JUMPED])
    car.air_time_since_jump = float(data[CAR_AIR_TIME_SINCE_JUMP])
    car.flip_time = float(data[CAR_FLIP_TIME])
    car.flip_torque = data[CAR_FLIP_TORQUE]

    car.is_autoflipping = bool(data[CAR_IS_AUTOFLIPPING])
    car.autoflip_timer = float(data[CAR_AUTOFLIP_TIMER])
    car.autoflip_direction = float(data[CAR_AUTOFLIP_DIRECTION])

    car.physics = deserialize_physics_object(data[CAR_PHYSICS])

    return car


def serialize_physics_object(physics_object: PhysicsObject) -> np.ndarray[tuple[Literal[18]], np.dtype[np.float32]]:
    return np.concatenate([
        physics_object.position,
        physics_object.linear_velocity,
        physics_object.angular_velocity,
        physics_object.rotation_mtx.flatten()
    ], dtype=np.float32)


# Indices for serialized PhysicsObject:
PO_POSITION = slice(0, 3)
PO_LINEAR_VELOCITY = slice(3, 6)
PO_ANGULAR_VELOCITY = slice(6, 9)
PO_ROTATION_MTX = slice(9, 18)


def deserialize_physics_object(data: np.ndarray[tuple[Literal[18]], np.dtype[np.float32]]) -> PhysicsObject:
    assert data.shape == (18,)
    po = PhysicsObject()
    po.position = data[PO_POSITION]
    po.linear_velocity = data[PO_LINEAR_VELOCITY]
    po.angular_velocity = data[PO_ANGULAR_VELOCITY]
    po.rotation_mtx = data[PO_ROTATION_MTX].reshape(3, 3)

    return po


def serialize_scoreboard(scoreboard: ScoreboardInfo):
    return np.array([
        scoreboard.game_timer_seconds,
        scoreboard.kickoff_timer_seconds,
        scoreboard.blue_score,
        scoreboard.orange_score,
        int(scoreboard.go_to_kickoff),
        int(scoreboard.is_over),
    ], dtype=np.float32)


# Indices for serialized ScoreboardInfo:
SB_GAME_TIMER_SECONDS = 0
SB_KICKOFF_TIMER_SECONDS = 1
SB_BLUE_SCORE = 2
SB_ORANGE_SCORE = 3
SB_GO_TO_KICKOFF = 4
SB_IS_OVER = 5


def deserialize_scoreboard(data: np.ndarray):
    assert data.shape == (6,)
    return ScoreboardInfo(
        game_timer_seconds=float(data[SB_GAME_TIMER_SECONDS]),
        kickoff_timer_seconds=float(data[SB_KICKOFF_TIMER_SECONDS]),
        blue_score=int(data[SB_BLUE_SCORE]),
        orange_score=int(data[SB_ORANGE_SCORE]),
        go_to_kickoff=bool(data[SB_GO_TO_KICKOFF]),
        is_over=bool(data[SB_IS_OVER]),
    )


def main():
    class RenameAgentMutator(StateMutator):
        def apply(self, state: StateType, shared_info: Dict[str, Any]) -> None:
            state.cars = {i: v for i, v in enumerate(state.cars.values())}

    transition_engine = RocketSimEngine()
    env = RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(), KickoffMutator(), RenameAgentMutator()),
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction()),
        reward_fn=GoalReward(),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
    )

    obs = env.reset()
    states = []
    while True:
        state = transition_engine.state
        s = serialize(state)
        states.append(s)
        d = deserialize(s)

        assert d.tick_count == state.tick_count
        assert d.goal_scored == state.goal_scored
        assert d.config.gravity == state.config.gravity
        assert d.config.boost_consumption == state.config.boost_consumption
        assert d.config.dodge_deadzone == state.config.dodge_deadzone
        assert np.allclose(d.ball.position, state.ball.position)
        assert np.allclose(d.ball.linear_velocity, state.ball.linear_velocity)
        assert np.allclose(d.ball.angular_velocity, state.ball.angular_velocity)
        assert np.allclose(d.ball.rotation_mtx, state.ball.rotation_mtx)
        assert np.allclose(d.boost_pad_timers, state.boost_pad_timers)
        assert len(d.cars) == len(state.cars)
        car_check = {}
        for i, v in state.cars.items():
            for attr in v.__slots__:
                if attr.startswith("_"):
                    continue
                val = getattr(v, attr)
                other_val = getattr(d.cars[i], attr)
                if isinstance(val, np.ndarray):
                    car_check[attr] = np.allclose(getattr(d.cars[i], attr), val)
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
        if any(terminated.values()):
            env.reset()


if __name__ == '__main__':
    main()
