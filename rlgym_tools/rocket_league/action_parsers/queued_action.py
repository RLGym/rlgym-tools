from typing import Dict, Any, Tuple, List

import numpy as np
from rlgym.api import ActionParser, AgentID, ActionType, EngineActionType, ActionSpaceType
from rlgym.rocket_league.api import GameState


class QueuedAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, Tuple[AgentID, np.ndarray]]):
    def __init__(self, parser: ActionParser, action_queue_size: int = 1):
        """
        QueuedAction maintains a queue of actions to execute and adds parsed actions to the queue.

        :param parser: the action parser to parse actions that are then added to the queue.
        """
        super().__init__()
        self.parser = parser
        self.action_queue_size = action_queue_size
        self.action_queue = {}
        self.is_initial = True

    def get_action_space(self, agent: AgentID) -> ActionSpaceType:
        return self.parser.get_action_space(agent)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.parser.reset(agents, initial_state, shared_info)
        self.action_queue = {k: [] for k in initial_state.cars.keys()}
        self.is_initial = True
        shared_info["action_queue"] = self.action_queue

    def parse_actions(self, actions: Dict[AgentID, ActionType], state: GameState, shared_info: Dict[str, Any]) \
            -> Dict[AgentID, EngineActionType]:
        parsed_actions = self.parser.parse_actions(actions, state, shared_info)
        returned_actions = {}
        if self.is_initial:
            for agent, action in parsed_actions.items():
                self.action_queue[agent] = [action] * self.action_queue_size
            self.is_initial = False
        for agent, action in parsed_actions.items():
            self.action_queue[agent].append(action)
            returned_actions[agent] = self.action_queue[agent].pop(0)
        shared_info["action_queue"] = self.action_queue
        return returned_actions
