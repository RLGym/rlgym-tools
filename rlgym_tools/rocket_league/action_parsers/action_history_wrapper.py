from typing import Tuple, Dict, Any, List

import numpy as np
from rlgym.api import ActionParser, AgentID, ActionType, StateType, EngineActionType, ActionSpaceType
from rlgym.rocket_league.api import GameState


class ActionHistoryWrapper(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, Tuple[AgentID, np.ndarray]]):
    def __init__(self, action_parser: ActionParser, ticks_to_remember: int = 8):
        super().__init__()
        self.action_parser = action_parser
        self.action_history = {}
        self.ticks_to_remember = ticks_to_remember

    def get_action_space(self, agent: AgentID) -> ActionSpaceType:
        return self.action_parser.get_action_space(agent)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.action_parser.reset(agents, initial_state, shared_info)
        self.action_history = {agent: np.zeros((0, 8)) for agent in agents}

    def parse_actions(self, actions: Dict[AgentID, ActionType], state: GameState, shared_info: Dict[str, Any]) \
            -> Dict[AgentID, EngineActionType]:
        parsed_actions = self.action_parser.parse_actions(actions, state, shared_info)
        for agent, action in parsed_actions.items():
            # Action has shape (ticks, 8)
            history = self.action_history[agent]
            if len(history) < self.ticks_to_remember:
                self.action_history[agent] = np.vstack((history, action))
            elif len(action) >= self.ticks_to_remember:
                self.action_history[agent] = action[-self.ticks_to_remember:]
            else:
                history[:-len(action)] = history[len(action):]
                history[-len(action):] = action
        shared_info["action_history"] = self.action_history
        return parsed_actions
