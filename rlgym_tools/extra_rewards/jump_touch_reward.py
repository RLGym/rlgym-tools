from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """

    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
            ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0
