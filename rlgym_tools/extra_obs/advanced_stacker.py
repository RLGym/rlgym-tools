import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from collections import deque


class AdvancedStacker(ObsBuilder):
    """
    Alternative observation to AdvancedObs that stacks AdvancedObs of the same info as in AdvancedObs and also actions
    that led into those observations.

    :param stack_size: number of frames to stack
    """

    def __init__(self, stack_size: int = 15):
        super().__init__()
        self.POS_STD = 6000
        self.VEL_STD = 3000
        self.ANG_STD = 5.5
        self.default_action = [0, 0, 0, 0, 0, 0, 0, 0]
        self.stack_size = stack_size
        self.action_stack = [deque([], maxlen=self.stack_size) for _ in range(66)]
        for i in range(len(self.action_stack)):
            self.blank_stack(i)

    def blank_stack(self, index: int) -> None:
        for _ in range(self.stack_size):
            self.action_stack[index].appendleft(self.default_action)

    def add_action_to_stack(self, new_action: np.ndarray, index: int):
        self.action_stack[index].appendleft(new_action)

    def reset(self, initial_state: GameState):
        for p in initial_state.players:
            self.blank_stack(p.car_id)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self.add_action_to_stack(previous_action, player.car_id)

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [
            ball.position / self.POS_STD,
            ball.linear_velocity / self.VEL_STD,
            ball.angular_velocity / self.ANG_STD,
            previous_action,
            pads,
        ]

        obs.extend(list(self.action_stack[player.car_id]))

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend(
                [
                    (other_car.position - player_car.position) / self.POS_STD,
                    (other_car.linear_velocity - player_car.linear_velocity)
                    / self.VEL_STD,
                ]
            )

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend(
            [
                rel_pos / self.POS_STD,
                rel_vel / self.VEL_STD,
                player_car.position / self.POS_STD,
                player_car.forward(),
                player_car.up(),
                player_car.linear_velocity / self.VEL_STD,
                player_car.angular_velocity / self.ANG_STD,
                [
                    player.boost_amount,
                    int(player.on_ground),
                    int(player.has_flip),
                    int(player.is_demoed),
                ],
            ]
        )

        return player_car
