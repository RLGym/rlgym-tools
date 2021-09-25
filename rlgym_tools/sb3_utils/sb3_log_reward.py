import os
from typing import Tuple, Optional, List

import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import CombinedReward
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger


class SB3LogReward(RewardFunction):
    """
    Simple reward function for logging individual rewards to a custom Logger.
    """
    def __init__(self, logger: Logger, reward_function: RewardFunction):
        super().__init__()
        self.logger = logger
        self.reward_function = reward_function
        self.reward_sum = 0
        self.episode_steps = 0
        self.global_steps = 0

    def reset(self, initial_state: GameState):
        if self.episode_steps > 0:
            rew_fn_type = type(self.reward_function)
            mean_reward = self.reward_sum / self.episode_steps
            if rew_fn_type.__str__ is not object.__str__:
                self.logger.record(f"{self.reward_function}/ep_rew_mean", mean_reward)
            else:
                self.logger.record(f"{rew_fn_type.__name__}/ep_rew_mean", mean_reward)
            self.logger.dump(self.global_steps)
            self.reward_sum = 0
            self.episode_steps = 0
            self.global_steps += 1

        self.reward_function.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        self.reward_sum += rew
        self.episode_steps += 1
        return rew

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        self.reward_sum += rew
        self.episode_steps += 1
        return rew

class SB3CombinedLogReward(CombinedReward):

    index_tracker = 0 # So we can uniquely identify each instance

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward. Will also log the weighted rewards to
        the model's logger if a SB3CombinedLogRewardCallback is provided to the
        learner.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__(reward_functions,reward_weights)
        self.index = SB3CombinedLogReward.index_tracker
        SB3CombinedLogReward.index_tracker += 1

        # make sure there is a folder to dump to
        os.makedirs('bin/combinedlogfiles', exist_ok=True)
        self.dumpfile = f'bin/combinedlogfiles/{self.index}.txt'

        # clean out any files that are left over from previous runs
        if self.index == 0:
            for file in os.listdir('bin/combinedlogfiles'):
                if '.txt' in file:
                    os.remove('bin/combinedlogfiles/'+file)


        # initiates the array that will store the rewards
        self.returns = np.zeros(len(self.reward_functions))

    def reset(self, initial_state: GameState):
        self.returns = np.zeros(len(self.reward_functions))
        super().reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.returns += [a*b for a, b in zip(rewards, self.reward_weights)] # store the rewards

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.returns += [a*b for a, b in zip(rewards, self.reward_weights)] # store the rewards

        # write the rewards to file and reset
        with open(self.dumpfile, 'a') as f:
            # in a list so we can eval to get it back easily
            f.write('\n' + str(list(self.returns)))
        self.returns = np.zeros(len(self.reward_functions))

        return float(np.dot(self.reward_weights, rewards))


class SB3CombinedLogRewardCallback(BaseCallback):
    def __init__(self, rew_names:List[str]):
        """
        Callback to log the data from a SB3CombinedLogReward to the
        same log as the model.

        :param rew_names: List of names that the logger will use for
        each reward.
        """
        super().__init__()
        self.rew_names = rew_names

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        returns = []
        # get all returns out of the files
        for file in os.listdir('bin/combinedlogfiles'):
            if '.txt' in file:
                with open('bin/combinedlogfiles/'+file, 'r') as f:
                    for line in f:
                        if line != '\n':
                            returns.append(eval(line.strip()))
                # empty the file ready for the next rollout
                with open('bin/combinedlogfiles/'+file, 'w') as f:
                    f.write('') 

        returns = np.array(returns)
        # use as many names as provided, then call them by their index after that
        names = [self.rew_names[i] if i < len(self.rew_names) else f'reward_{i}' for i in range(returns.shape[1])]

        # log each reward
        for n, name in enumerate(names):
            self.model.logger.record('rewards/'+name, np.mean(returns[:, n]))
