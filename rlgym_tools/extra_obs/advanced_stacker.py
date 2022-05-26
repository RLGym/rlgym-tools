import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder


class AdvancedStacker(ObsBuilder):
    """
    Alternative observation to AdvancedObs. action_stacks past stack_size actions and appends them to AdvancedObs. If there were no previous actions, zeros are assumed as previous actions.

    :param stack_size: number of frames to stack
    """

    def __init__(self, stack_size: int = 15):
        super().__init__()
        self.POS_STD = 6000
        self.VEL_STD = 3000
        self.ANG_STD = 5.5
        self.default_action = np.zeros(common_values.NUM_ACTIONS)
        self.stack_size = stack_size
        self.action_stacks = {}
        self.action_size = self.default_action.shape[0]

    def blank_stack(self, car_id: int) -> None:

    def add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.action_stacks[car_id]
        stack[self.action_size:] = stack[:-self.action_size]
        stack[:self.action_size] = new_action

    def reset(self, initial_state: GameState):
        self.action_stacks = {}
        for p in initial_state.players:
            self.action_stacks[p.car_id] = np.concatenate([default_action] * stack_size
            

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

        obs.extend(list(self.action_stacks[player.car_id]))

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
