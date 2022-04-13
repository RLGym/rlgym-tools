from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward

from rlgym.utils.gamestates import GameState, PlayerData

from numpy import ndarray


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """
    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.vel_dir_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        return reward
