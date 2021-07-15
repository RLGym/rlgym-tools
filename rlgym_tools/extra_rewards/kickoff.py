import numpy as np

from rlgym.utils import RewardFunction, TerminalCondition, math
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, EventReward
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_rewards.diff_reward import DiffReward


class KickoffCondition(TerminalCondition):
    def __init__(self, fps):
        super().__init__()
        self.no_touch = NoTouchTimeoutCondition(5 * round(fps))
        self.goal = GoalScoredCondition()

    def reset(self, initial_state: GameState):
        self.no_touch.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        if self.no_touch.is_terminal(current_state) or self.goal.is_terminal(current_state):
            return True
        n_leavers = 0  # Counts how many people are going away from ball
        for player in current_state.players:
            if player.team_num == BLUE_TEAM:
                ball = current_state.ball
                car_data = player.car_data
            else:
                ball = current_state.inverted_ball
                car_data = player.inverted_car_data

            if ball.position[1] < car_data.position[1] - 100 and ball.linear_velocity[1] < car_data.linear_velocity[1]:
                return True

            if math.scalar_projection(player.car_data.linear_velocity,
                                      current_state.ball.position - player.car_data.position) < -100:
                n_leavers += 1

        return n_leavers == len(current_state.players)


class KickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.vel_reward = DiffReward(VelocityPlayerToBallReward())
        self.touch_reward = EventReward(touch=1)

    def reset(self, initial_state: GameState):
        self.vel_reward.reset(initial_state)
        self.touch_reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        args = (player, state, previous_action)
        return self.vel_reward.get_reward(*args) + self.touch_reward.get_reward(*args)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            ball_y = state.ball.position[1]
            player_y = player.car_data.position[1]
        else:
            ball_y = state.inverted_ball.position[1]
            player_y = player.inverted_car_data.position[1]

        if ball_y < player_y:
            return -10

        return 0

        # rew = 0
        # for p in state.players:
        #     if player.team_num == BLUE_TEAM:
        #         ball_y = state.ball.position[1]
        #         player_y = player.car_data.position[1]
        #     else:
        #         ball_y = state.inverted_ball.position[1]
        #         player_y = player.inverted_car_data.position[1]
        #
        #     if ball_y < player_y:
        #         if p.team_num == player.team_num:
        #             rew -= 10
        #         else:
        #             rew += 1
        #
        # return rew
