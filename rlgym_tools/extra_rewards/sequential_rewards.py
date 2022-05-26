from rlgym.utils.reward_functions import RewardFunction

from rlgym.utils.gamestates import GameState, PlayerData

from numpy import ndarray

class SequentialRewards(RewardFunction):
    """ 
    A simple reward class that allows you to transition from one reward to the next at set intervals.
    Example: rewards, step_requirements = [reward1, reward2, reward3, etc], [10_000_000, 20_000_000, 30_000_000, etc]
    my_rewards = SequentialRewards(rewards, step_requirements)   
    """
    def __init__(self, rewards: list, steps: list):
        super().__init__()
        self.rewards_list = rewards
        self.step_counts = steps
        self.step_count = 0
        self.step_index = 0
        assert len(self.rewards_list) == len(self.step_counts)

    def reset(self, initial_state: GameState):
        for rew in self.rewards_list:
            rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        if self.step_index < len(self.step_counts) and self.step_count > self.step_counts[self.step_index]:
            self.step_index += 1

        self.step_count += 1
        return self.rewards_list[self.step_index].get_reward(player, state, previous_action)
