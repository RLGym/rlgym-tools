import json
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

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None,
            file_location: str = 'combinedlogfiles'
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward. Will also log the weighted rewards to
        the model's logger if a SB3CombinedLogRewardCallback is provided to the
        learner.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        :param file_location: The path to the directory that will be used to
        transfer reward info
        """
        super().__init__(reward_functions, reward_weights)

        # Make sure there is a folder to dump to
        os.makedirs(file_location, exist_ok=True)
        self.file_location = f'{file_location}/rewards.txt'
        self.lockfile = f'{file_location}/reward_lock'

        # Initiates the array that will store the episode totals
        self.returns = np.zeros(len(self.reward_functions))

        # Obtain the lock
        while True:
            try:
                open(self.lockfile, 'x')
                break
            except FileExistsError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f'Error obtaining lock in SB3CombinedLogReward.__init__:\n{e}')

        # Empty the file by opening in w mode
        with open(self.file_location, 'w') as f:
            pass

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

    def reset(self, initial_state: GameState):
        self.returns = np.zeros(len(self.reward_functions))
        super().reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]  # store the rewards

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        # Add the rewards to the cumulative totals with numpy broadcasting
        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]

        # Obtain the lock
        while True:
            try:
                open(self.lockfile, 'x')
                break
            except FileExistsError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f'Error obtaining lock in SB3CombinedLogReward.get_final_reward:\n{e}')

        # Write the rewards to file and reset
        with open(self.file_location, 'a') as f:
            f.write('\n' + json.dumps(self.returns.tolist()))

        # reset the episode totals
        self.returns = np.zeros(len(self.reward_functions))

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

        return float(np.dot(self.reward_weights, rewards))


class SB3CombinedLogRewardCallback(BaseCallback):
    def __init__(self, reward_names: List[str], file_location: str = 'combinedlogfiles'):
        """
        Callback to log the data from a SB3CombinedLogReward to the
        same log as the model.

        :param reward_names: List of names that the logger will use for
        each reward.
        :param file_location: The path to the directory that will be used to
        transfer reward info
        """
        super().__init__()
        self.reward_names = reward_names
        self.file_location = file_location + '/rewards.txt'
        self.lockfile = file_location + '/reward_lock'

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        returns = []

        # Obtain the lock
        while True:
            try:
                open(self.lockfile, 'x')
                break
            except FileExistsError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f'Error obtaining lock in SB3CombinedLogRewardCallback._on_rollout_end:\n{e}')

        # Read the file into returns
        with open(self.file_location, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        line = json.loads(line)
                        returns.append(line)

                    except Exception as e:
                        print(f'Exception loading line {line}:\n\t{e}')

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

        # Make returns into a numpy array so we can make use of numpy features
        returns = np.array(returns)

        # Log each reward
        for n in range(returns.shape[1]):
            try:
                name = self.reward_names[n]
            except IndexError:
                name = f'reward_{n}'
            self.model.logger.record_mean('rewards/' + name, np.mean(returns[:, n]))
