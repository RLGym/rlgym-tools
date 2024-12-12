from typing import List, Dict, Any

from rlgym.api import DoneCondition, AgentID, StateType
from rlgym.rocket_league.api import GameState


class CarOnGroundCondition(DoneCondition[AgentID, GameState]):
    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        return {agent: state.cars[agent].on_ground for agent in agents}
