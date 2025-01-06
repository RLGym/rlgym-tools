from typing import Dict, Any, List

import numpy as np
from rlgym.api import TransitionEngine
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim import RocketSimEngine


class MultiEnvEngine(TransitionEngine[tuple, List[GameState], np.ndarray]):
    """
    MultiEnvEngine is a wrapper around multiple RocketSimEngine instances. It allows for multiple environments to be
    run in "parallel". While the environments are not actually run in parallel, it can still be useful since performing
    batched model inference is usually faster than performing inference on a single instance at a time.
    """

    def __init__(self, num_envs: int):
        # TODO use the parallel step method from RocketSim bindings
        self.envs = [
            RocketSimEngine()
            for _ in range(num_envs)
        ]

    @property
    def agents(self) -> List[tuple]:
        agents = []
        for i, env in enumerate(self.envs):
            for agent in env.agents:
                agents.append((i, agent))
        return agents

    @property
    def max_num_agents(self) -> int:
        return sum(env.max_num_agents for env in self.envs)

    @property
    def state(self) -> List[GameState]:
        return [env.state for env in self.envs]

    @property
    def config(self) -> Dict[str, Any]:
        configs = [env.config for env in self.envs]
        keys = configs[0].keys()
        return {key: [config[key] for config in configs] for key in keys}

    @config.setter
    def config(self, value: Dict[str, Any]):
        for i in range(len(self.envs)):
            # Make sure we use RocketSimEngine's config.setter
            self.envs[i].config = {  # noqa setter not implemented
                key: value[key][i]
                for key in value.keys()
            }

    def step(self, actions: Dict[tuple, np.ndarray], shared_info: Dict[str, Any]) -> List[GameState]:
        grouped_actions = [{} for _ in range(len(self.envs))]
        for (i, agent), action in actions.items():
            grouped_actions[i][agent] = action
        states = []
        for i, env in enumerate(self.envs):
            state = env.step(grouped_actions[i], shared_info)
            states.append(state)
        return states

    def create_base_state(self) -> List[GameState]:
        return [env.create_base_state() for env in self.envs]

    def set_state(self, desired_state: List[GameState], shared_info: Dict[str, Any]) -> List[GameState]:
        for i, env in enumerate(self.envs):
            env.set_state(desired_state[i], shared_info)
        return self.state

    def close(self) -> None:
        for env in self.envs:
            env.close()
