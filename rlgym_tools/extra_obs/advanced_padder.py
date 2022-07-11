from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from typing import Any, List
from rlgym.utils import common_values
import numpy as np
import math


class AdvancedObsPadder(ObsBuilder):
    """adds 0 padding to accommodate differing numbers of agents"""
    def __init__(self, team_size=3, expanding=False):
        super().__init__()
        self.team_size = team_size
        self.POS_STD = 2300
        self.ANG_STD = math.pi
        self.expanding = expanding

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []
        ally_count = 0
        enemy_count = 0
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
                ally_count += 1
                if ally_count > self.team_size-1:
                    continue
            else:
                team_obs = enemies
                enemy_count += 1
                if enemy_count > self.team_size:
                    continue
            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        while ally_count < self.team_size-1:
            self._add_dummy(allies)
            ally_count += 1

        while enemy_count < self.team_size:
            self._add_dummy(enemies)
            enemy_count += 1

        obs.extend(allies)
        obs.extend(enemies)
        if self.expanding:
            return np.expand_dims(np.concatenate(obs), 0)
        return np.concatenate(obs)

    def _add_dummy(self, obs: List):
        obs.extend([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [0, 0, 0, 0, 0]])
        obs.extend([np.zeros(3), np.zeros(3)])

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed),
             int(player.has_jump)]])

        return player_car


if __name__ == "__main__":
    pass
