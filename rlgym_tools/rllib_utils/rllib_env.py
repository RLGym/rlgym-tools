from typing import Tuple
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict
from rlgym.gym import Gym

class RLLibEnv(MultiAgentEnv):
    def __init__(self, env: Gym):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._agent_ids = [i for i in range(env._match.agents)]

    @override(MultiAgentEnv)
    def reset(self) -> MultiAgentDict:
        observations = self.env.reset()
        return {i: obs for i, obs in enumerate(observations)}

    @override(MultiAgentEnv)
    def step(self, action_dict: MultiAgentDict) -> \
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        action_array = [val for val in action_dict.values()]
        observations, rewards, done, info = self.env.step(action_array)

        obs_dict = {}
        rew_dict = {}
        done_dict = {"__all__": done}
        info_dict = {}
        for i, (obs, rew) in enumerate(zip(observations, rewards)):
            obs_dict[i] = obs
            rew_dict[i] = rew
            info_dict[i] = info
        return obs_dict, rew_dict, done_dict, info_dict

    @override(MultiAgentEnv)
    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        return {agent_id:self.action_space.sample() for agent_id in self._agent_ids}
