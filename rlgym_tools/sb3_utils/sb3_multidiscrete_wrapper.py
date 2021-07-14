import numpy as np
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs, VecEnv


class SB3MultiDiscreteWrapper(VecEnvWrapper):
    """
    Simply converts env such that action space is MultiDiscrete instead of Box (basically KBM).
    """
    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        self.action_space = MultiDiscrete((3, 3, 3, 3, 3, 2, 2, 2))  # TODO allow for more granular outputs

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.copy(actions)
        actions[..., :5] -= 1
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
