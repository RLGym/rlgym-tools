from typing import List, Dict, Any

from rlgym.api import SharedInfoProvider, AgentID, StateType
from rlgym.rocket_league.api import GameState

from rlgym_tools.rocket_league.misc.serialize import serialize


class SerializedProvider(SharedInfoProvider[AgentID, GameState]):
    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        return shared_info

    def set_state(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        serialized_state = serialize(initial_state)
        shared_info["serialized_state"] = serialized_state
        return shared_info

    def step(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        serialized_state = serialize(state)
        shared_info["serialized_state"] = serialized_state
        return shared_info
