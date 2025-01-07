from typing import Literal

import numpy as np
from rlgym.rocket_league.api import GameState, Car, PhysicsObject, GameConfig
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo


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
    elif isinstance(obj, ReplayFrame):
        return serialize_replay_frame(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def deserialize(obj):
    if isinstance(obj, np.ndarray):
        if obj.shape == (18,):
            return deserialize_physics_object(obj)
        elif obj.shape == (45,):
            return deserialize_car(obj)
        elif obj.shape == (3,):
            return deserialize_config(obj)
        elif obj.shape == (6,):
            return deserialize_scoreboard(obj)
        else:
            try:
                return deserialize_game_state(obj)
            except AssertionError:
                return deserialize_replay_frame(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


# Indices for serialized GameState:
GS_TICK_COUNT = 0
GS_GOAL_SCORED = 1
GS_CONFIG = slice(2, 5)
GS_BALL = slice(5, 23)
GS_BOOST_PAD_TIMERS = slice(23, 23 + len(BOOST_LOCATIONS))
GS_CARS = slice(23 + len(BOOST_LOCATIONS), None)
GS_CAR_LENGTH = 46  # 1 for agent ID, 45 for car


def serialize_game_state(game_state: GameState, aid_to_num=None) -> np.ndarray:
    # Make sure agent IDs are compatible with serialization
    if aid_to_num is None:
        aid_to_num = {}
        for agent_id in sorted(game_state.cars.keys()):
            if isinstance(agent_id, (float, int)):
                aid_to_num[agent_id] = agent_id
            else:
                aid_to_num[agent_id] = len(aid_to_num)

    s = np.zeros(2 + 3 + 18 + len(BOOST_LOCATIONS) + len(game_state.cars) * GS_CAR_LENGTH,
                 dtype=np.float32)
    s[GS_TICK_COUNT] = game_state.tick_count
    s[GS_GOAL_SCORED] = game_state.goal_scored
    s[GS_CONFIG] = serialize_config(game_state.config)
    s[GS_BALL] = serialize_physics_object(game_state.ball)
    s[GS_BOOST_PAD_TIMERS] = game_state.boost_pad_timers
    for i, (aid, car) in enumerate(game_state.cars.items()):
        start = GS_CARS.start + i * GS_CAR_LENGTH
        end = start + GS_CAR_LENGTH
        s[start] = aid_to_num[aid]
        s[start + 1:end] = serialize_car(car, aid_to_num)
    return s


def deserialize_game_state(data: np.ndarray) -> GameState:
    assert (len(data) - GS_CARS.start) % GS_CAR_LENGTH == 0, f"Invalid data length: {len(data)}"
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
    for i in range(0, len(rest), GS_CAR_LENGTH):
        if np.all(rest[i:i + GS_CAR_LENGTH] == 0):  # Padding
            break
        agent_id = int(rest[i])
        car = deserialize_car(rest[i + 1:i + GS_CAR_LENGTH])
        cars[agent_id] = car
    game_state.cars = cars

    return game_state


# Indices for serialized GameConfig:
CONFIG_GRAVITY = 0
CONFIG_BOOST_CONSUMPTION = 1
CONFIG_DODGE_DEADZONE = 2


def serialize_config(config: GameConfig) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]:
    s = np.zeros(3, dtype=np.float32)
    s[CONFIG_GRAVITY] = config.gravity
    s[CONFIG_BOOST_CONSUMPTION] = config.boost_consumption
    s[CONFIG_DODGE_DEADZONE] = config.dodge_deadzone
    return s


def deserialize_config(data: np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]) -> GameConfig:
    assert data.shape == (3,)
    config = GameConfig()
    config.gravity = float(data[CONFIG_GRAVITY])
    config.boost_consumption = float(data[CONFIG_BOOST_CONSUMPTION])
    config.dodge_deadzone = float(data[CONFIG_DODGE_DEADZONE])

    return config


# Indices for serialized Car:
CAR_TEAM_NUM = 0
CAR_HITBOX_TYPE = 1
CAR_BALL_TOUCHES = 2
CAR_BUMP_VICTIM_ID = 3

CAR_DEMO_RESPAWN_TIMER = 4
CAR_WHEELS_WITH_CONTACT = range(5, 9)
CAR_SUPERSONIC_TIME = 9
CAR_BOOST_AMOUNT = 10
CAR_BOOST_ACTIVE_TIME = 11
CAR_HANDBRAKE = 12

CAR_HAS_JUMPED = 13
CAR_IS_HOLDING_JUMP = 14
CAR_IS_JUMPING = 15
CAR_JUMP_TIME = 16

CAR_HAS_FLIPPED = 17
CAR_HAS_DOUBLE_JUMPED = 18
CAR_AIR_TIME_SINCE_JUMP = 19
CAR_FLIP_TIME = 20
CAR_FLIP_TORQUE = slice(21, 24)

CAR_IS_AUTOFLIPPING = 24
CAR_AUTOFLIP_TIMER = 25
CAR_AUTOFLIP_DIRECTION = 26

CAR_PHYSICS = slice(27, 27 + 18)


def serialize_car(car: Car, bump_victim_map=None) -> np.ndarray[tuple[Literal[42]], np.dtype[np.float32]]:
    bump_victim = car.bump_victim_id
    if bump_victim is None:
        bump_victim = -1
    else:
        bump_victim_map = bump_victim_map or {}
        if not isinstance(bump_victim, (int, float)):
            bump_victim = bump_victim_map.get(bump_victim, -2)
        assert bump_victim != -1, "Invalid bump victim ID (-1 is reserved for None)"

    s = np.zeros(45, dtype=np.float32)
    s[CAR_TEAM_NUM] = car.team_num
    s[CAR_HITBOX_TYPE] = car.hitbox_type
    s[CAR_BALL_TOUCHES] = car.ball_touches
    s[CAR_BUMP_VICTIM_ID] = bump_victim
    s[CAR_DEMO_RESPAWN_TIMER] = car.demo_respawn_timer
    s[CAR_WHEELS_WITH_CONTACT] = car.wheels_with_contact
    s[CAR_SUPERSONIC_TIME] = car.supersonic_time
    s[CAR_BOOST_AMOUNT] = car.boost_amount
    s[CAR_BOOST_ACTIVE_TIME] = car.boost_active_time
    s[CAR_HANDBRAKE] = car.handbrake
    s[CAR_HAS_JUMPED] = car.has_jumped
    s[CAR_IS_HOLDING_JUMP] = car.is_holding_jump
    s[CAR_IS_JUMPING] = car.is_jumping
    s[CAR_JUMP_TIME] = car.jump_time
    s[CAR_HAS_FLIPPED] = car.has_flipped
    s[CAR_HAS_DOUBLE_JUMPED] = car.has_double_jumped
    s[CAR_AIR_TIME_SINCE_JUMP] = car.air_time_since_jump
    s[CAR_FLIP_TIME] = car.flip_time
    s[CAR_FLIP_TORQUE] = car.flip_torque
    s[CAR_IS_AUTOFLIPPING] = car.is_autoflipping
    s[CAR_AUTOFLIP_TIMER] = car.autoflip_timer
    s[CAR_AUTOFLIP_DIRECTION] = car.autoflip_direction
    s[CAR_PHYSICS] = serialize_physics_object(car.physics)
    return s


def deserialize_car(data: np.ndarray[tuple[Literal[42]], np.dtype[np.float32]]) -> Car:
    assert data.shape == (45,)
    car = Car()
    car.team_num = int(data[CAR_TEAM_NUM])
    car.hitbox_type = int(data[CAR_HITBOX_TYPE])
    car.ball_touches = int(data[CAR_BALL_TOUCHES])
    car.bump_victim_id = int(data[CAR_BUMP_VICTIM_ID]) if data[CAR_BUMP_VICTIM_ID] != -1 else None

    car.demo_respawn_timer = float(data[CAR_DEMO_RESPAWN_TIMER])
    car.wheels_with_contact = tuple(bool(data[i]) for i in CAR_WHEELS_WITH_CONTACT)  # noqa trust that length is correct
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


# Indices for serialized PhysicsObject:
PO_POSITION = slice(0, 3)
PO_LINEAR_VELOCITY = slice(3, 6)
PO_ANGULAR_VELOCITY = slice(6, 9)
PO_ROTATION_MTX = slice(9, 18)


def serialize_physics_object(physics_object: PhysicsObject) -> np.ndarray[tuple[Literal[18]], np.dtype[np.float32]]:
    s = np.zeros(18, dtype=np.float32)
    s[PO_POSITION] = physics_object.position
    s[PO_LINEAR_VELOCITY] = physics_object.linear_velocity
    s[PO_ANGULAR_VELOCITY] = physics_object.angular_velocity
    s[PO_ROTATION_MTX] = physics_object.rotation_mtx.flatten()
    return s


def deserialize_physics_object(data: np.ndarray[tuple[Literal[18]], np.dtype[np.float32]]) -> PhysicsObject:
    assert data.shape == (18,)
    po = PhysicsObject()
    po.position = data[PO_POSITION]
    po.linear_velocity = data[PO_LINEAR_VELOCITY]
    po.angular_velocity = data[PO_ANGULAR_VELOCITY]
    po.rotation_mtx = data[PO_ROTATION_MTX].reshape(3, 3)

    return po


# Indices for serialized ScoreboardInfo:
SB_GAME_TIMER_SECONDS = 0
SB_KICKOFF_TIMER_SECONDS = 1
SB_BLUE_SCORE = 2
SB_ORANGE_SCORE = 3
SB_GO_TO_KICKOFF = 4
SB_IS_OVER = 5


def serialize_scoreboard(scoreboard: ScoreboardInfo):
    s = np.zeros(6, dtype=np.float32)
    s[SB_GAME_TIMER_SECONDS] = scoreboard.game_timer_seconds
    s[SB_KICKOFF_TIMER_SECONDS] = scoreboard.kickoff_timer_seconds
    s[SB_BLUE_SCORE] = scoreboard.blue_score
    s[SB_ORANGE_SCORE] = scoreboard.orange_score
    s[SB_GO_TO_KICKOFF] = scoreboard.go_to_kickoff
    s[SB_IS_OVER] = scoreboard.is_over
    return s


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


# Indices for serialized ReplayFrame:
RF_SCOREBOARD = slice(0, 6)
RF_EPISODE_SECONDS_REMAINING = 6
RF_NEXT_SCORING_TEAM = 7
RF_WINNING_TEAM = 8
RF_NUM_PLAYERS = 9
RF_AGENT_IDS_START = 10
RF_ACTION_SIZE = 8


def serialize_replay_frame(replay_frame: ReplayFrame):
    # state: GameState
    # actions: Dict[int, np.ndarray]
    # update_age: Dict[int, float]
    # scoreboard: ScoreboardInfo
    # episode_seconds_remaining: float
    # next_scoring_team: Optional[int]
    # winning_team: Optional[int]
    aid_to_num = {}
    agent_ids = sorted(replay_frame.state.cars.keys())
    for agent_id in agent_ids:
        if isinstance(agent_id, (float, int)):
            aid_to_num[agent_id] = agent_id
        else:
            aid_to_num[agent_id] = len(aid_to_num)

    num_players = len(replay_frame.state.cars)
    update_ages = np.array([replay_frame.update_age[agent_id] for agent_id in agent_ids], dtype=np.float32)
    actions = np.array([replay_frame.actions[agent_id] for agent_id in agent_ids], dtype=np.float32)
    agent_ids = np.array([aid_to_num[agent_id] for agent_id in agent_ids], dtype=np.float32)

    serialized_game_state = serialize_game_state(replay_frame.state, aid_to_num)
    s = np.zeros(6 + 3 + 1 + num_players + num_players + num_players * RF_ACTION_SIZE + serialized_game_state.shape[0],
                 dtype=np.float32)
    s[RF_SCOREBOARD] = serialize_scoreboard(replay_frame.scoreboard)
    s[RF_EPISODE_SECONDS_REMAINING] = replay_frame.episode_seconds_remaining
    s[RF_NEXT_SCORING_TEAM] = replay_frame.next_scoring_team if replay_frame.next_scoring_team is not None else -1
    s[RF_WINNING_TEAM] = replay_frame.winning_team if replay_frame.winning_team is not None else -1
    s[RF_NUM_PLAYERS] = num_players

    i = RF_AGENT_IDS_START
    j = i + num_players
    s[i:j] = agent_ids
    i = j
    j = i + num_players
    s[i:j] = update_ages
    i = j
    j = i + num_players * RF_ACTION_SIZE
    s[i:j] = actions.flatten()
    i = j
    j = i + serialized_game_state.shape[0]
    s[i:j] = serialized_game_state
    return s


def deserialize_replay_frame(data: np.ndarray):
    scoreboard = deserialize_scoreboard(data[RF_SCOREBOARD])
    episode_seconds_remaining = float(data[RF_EPISODE_SECONDS_REMAINING])
    next_scoring_team = int(data[RF_NEXT_SCORING_TEAM]) if data[RF_NEXT_SCORING_TEAM] != -1 else None
    winning_team = int(data[RF_WINNING_TEAM]) if data[RF_WINNING_TEAM] != -1 else None
    num_players = int(data[RF_NUM_PLAYERS])
    k = RF_AGENT_IDS_START
    agent_ids = data[k:k + num_players]
    agent_ids = [int(agent_id) for agent_id in agent_ids]
    k += num_players
    update_ages = data[k:k + num_players]
    k += num_players
    actions = data[k:k + num_players * RF_ACTION_SIZE].reshape(num_players, RF_ACTION_SIZE)
    k += num_players * RF_ACTION_SIZE
    state = deserialize_game_state(data[k:k + GS_CARS.start + num_players * GS_CAR_LENGTH])
    # assert np.all(data[k + GS_CARS.start + num_players * GS_CAR_LENGTH:] == 0), "Invalid padding"

    return ReplayFrame(
        state=state,
        actions={agent_id: action for agent_id, action in zip(agent_ids, actions)},
        update_age={agent_id: update_age for agent_id, update_age in zip(agent_ids, update_ages)},
        scoreboard=scoreboard,
        episode_seconds_remaining=episode_seconds_remaining,
        next_scoring_team=next_scoring_team,
        winning_team=winning_team,
    )
