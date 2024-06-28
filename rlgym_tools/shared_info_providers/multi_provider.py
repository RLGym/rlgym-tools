from typing import List, Dict, Any

from rlgym.api import SharedInfoProvider, AgentID, StateType
from rlgym.rocket_league.api import GameState


class MultiProvider(SharedInfoProvider[AgentID, GameState]):
    """
    Wrapper to use multiple SharedInfoProviders at once.
    """

    def __init__(self, *providers: SharedInfoProvider):
        self.providers = providers

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        for provider in self.providers:
            shared_info = provider.create(shared_info)
        return shared_info

    def set_state(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        for provider in self.providers:
            shared_info = provider.set_state(agents, initial_state, shared_info)
        return shared_info

    def step(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        for provider in self.providers:
            shared_info = provider.step(agents, state, shared_info)
        return shared_info
