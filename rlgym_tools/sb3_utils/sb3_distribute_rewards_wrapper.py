import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState


class SB3DistributeRewardsWrapper(VecEnvWrapper):
    """
    Does the same thing as DistributeRewards, but as a SB3 wrapper to enable logging of rewards before distributing.
    """
    def __init__(self, venv: VecEnv, team_spirit=0.3):
        super().__init__(venv)
        self.team_spirit = team_spirit

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        last_state = None
        for i, info in enumerate(infos):
            state: GameState = info["state"]
            if state != last_state:
                blue_indices = []
                orange_indices = []
                for n, player in enumerate(state.players):
                    if player.team_num == BLUE_TEAM:
                        blue_indices.append(n + i)
                    else:
                        orange_indices.append(n + i)

                for team, opp in (blue_indices, orange_indices), (orange_indices, blue_indices):
                    rewards[team] = (1 - self.team_spirit) * rewards[team] \
                                    + self.team_spirit * rewards[team].mean() \
                                    - np.nan_to_num(rewards[opp].mean())  # In case of no opponent

                last_state = state

        return observations, rewards, dones, infos
