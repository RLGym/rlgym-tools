from numpy import ndarray
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    TouchBallReward,
    VelocityPlayerToBallReward,
)


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """

    def __init__(self, reward_if_ball_touched: bool = True):
        self.reward_if_ball_touched = reward_if_ball_touched
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()
        if self.reward_if_ball_touched:
            self.touch_ball_reward = TouchBallReward()

    def reset(self, initial_state: GameState):
        self.vel_dir_reward.reset(initial_state)
        if self.reward_if_ball_touched:
            self.touch_ball_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
            if self.reward_if_ball_touched:
                reward += self.touch_ball_reward.get_reward(
                    player, state, previous_action
                )
        return reward
