from typing import Generator

from rlgym.rocket_league.api import Car, PhysicsObject, GameState
from rlgym.rocket_league.common_values import BLUE_TEAM

from rlgym_tools.rocket_league.v1 import V1PhysicsObject, V1PlayerData, V1GameState


def convert_physics_object(po: PhysicsObject, copy: bool = True) -> V1PhysicsObject:
    pos = po.position
    quat = po.quaternion
    lin_vel = po.linear_velocity
    ang_vel = po.angular_velocity

    if copy:
        pos = pos.copy()
        quat = quat.copy()
        lin_vel = lin_vel.copy()
        ang_vel = ang_vel.copy()

    v1_po = V1PhysicsObject(
        position=pos,
        quaternion=quat,
        linear_velocity=lin_vel,
        angular_velocity=ang_vel,
    )

    return v1_po


def convert_car(car: Car, car_id: int = -1, copy: bool = True) -> V1PlayerData:
    pd = V1PlayerData()
    pd.car_id = car_id
    pd.team_num = car.team_num
    pd.match_goals = 0
    pd.match_saves = 0
    pd.match_shots = 0
    pd.match_demolishes = 0
    pd.boost_pickups = 0
    pd.is_demoed = car.is_demoed
    pd.on_ground = car.on_ground
    pd.ball_touched = car.ball_touches > 0
    pd.has_jump = not car.has_jumped
    pd.has_flip = car.has_flip
    pd.boost_amount = car.boost_amount
    pd.car_data = convert_physics_object(car.physics, copy=copy)
    pd.inverted_car_data = convert_physics_object(car.inverted_physics, copy=copy)

    return pd


def convert_game_state(state: GameState, copy: bool = True) -> V1GameState:
    v1_state = V1GameState()
    # v1_state.game_type = 0
    if state.goal_scored:
        if state.scoring_team == BLUE_TEAM:
            v1_state.blue_score += 1
        else:
            v1_state.orange_score += 1
    v1_state.last_touch = -1

    # Sort cars by (team, agent_id)
    sorted_cars = sorted(state.cars.items(), key=lambda x: x[0])
    blue_cars = []
    orange_cars = []
    for agent_id, car in sorted_cars:
        if car.team_num == BLUE_TEAM:
            blue_cars.append(car)
        else:
            orange_cars.append(car)
    blue_start_index = 1
    if len(blue_cars) <= 4 and len(orange_cars) <= 4:
        orange_start_index = 5
    else:
        orange_start_index = len(blue_cars) + 1

    v1_state.players = ([convert_car(car, car_id=blue_start_index + i, copy=copy)
                         for i, car in enumerate(blue_cars)]
                        + [convert_car(car, car_id=orange_start_index + i, copy=copy)
                           for i, car in enumerate(orange_cars)])

    v1_state.ball = convert_physics_object(state.ball, copy=copy)
    v1_state.inverted_ball = convert_physics_object(state.inverted_ball, copy=copy)

    v1_state.boost_pads[:] = state.boost_pad_timers == 0
    v1_state.inverted_boost_pads[:] = state.inverted_boost_pad_timers == 0

    return v1_state


def convert_game_states(copy: bool = True) -> Generator[GameState, V1GameState, None]:
    # Slightly more accurate as it tracks last touch, demos, goals and boost pickups.
    # Still no saves and shots though.
    # Usage:
    # game_state_converter = convert_game_states()
    # next(game_state_converter)  # Start the generator
    # Do these two lines for each subsequent state:
    #   game_state_converter.send(state)  # Send state
    #   v1_state = next(game_state_converter)
    latest_touch = None
    blue_goals = 0
    orange_goals = 0
    prev_boost_amounts = {}
    while True:
        state: GameState = yield
        v1_state = convert_game_state(state, copy=copy)
        cars = sorted(state.cars.items(), key=lambda x: x[0])
        for (agent_id, car), player_data in zip(cars, v1_state.players):
            # Demos
            if car.bump_victim_id is not None and state.cars[car.bump_victim_id].is_demoed:
                player_data.match_demolishes += 1

            # Last touch for goal tracking
            if car.ball_touches > 0:
                latest_touch = player_data.car_id

            # Boost pickups
            if car.is_demoed:
                prev_boost_amounts.pop(agent_id, None)
            else:
                if agent_id in prev_boost_amounts:
                    diff = car.boost_amount - prev_boost_amounts[agent_id]
                    if diff > 0:
                        player_data.boost_pickups += 1
                prev_boost_amounts[agent_id] = car.boost_amount

        # Goals
        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                blue_goals += 1
            else:
                orange_goals += 1
            for player_data in v1_state.players:
                if player_data.car_id == latest_touch:
                    player_data.match_goals += 1

        v1_state.blue_score = blue_goals
        v1_state.orange_score = orange_goals

        yield v1_state
