from typing import Tuple, Dict, Any, List

import numpy as np
from rlgym.api import ActionParser, AgentID, ActionType, StateType, EngineActionType, ActionSpaceType
from rlgym.rocket_league.api import GameState


class DelayedAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, Tuple[AgentID, np.ndarray]]):
    """
    DelayedAction delays all actions by a specified number of ticks.
    The queue is put into the shared_info dictionary under the key "delayed_actions".
    The index of the queue indicates the number of ticks until the action is executed.
    """

    def __init__(self, action_parser: ActionParser, delay_ticks: int = 1):
        self.action_parser = action_parser
        self.delay_ticks = delay_ticks
        self.delayed_actions = None

    def get_action_space(self, agent: AgentID) -> ActionSpaceType:
        return self.action_parser.get_action_space(agent)

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.action_parser.reset(agents, initial_state, shared_info)
        self.delayed_actions = {agent: np.zeros((self.delay_ticks, 8)) for agent in agents}
        shared_info["delayed_actions"] = self.delayed_actions

    def parse_actions(self, actions: Dict[AgentID, ActionType], state: StateType, shared_info: Dict[str, Any]) \
            -> Dict[AgentID, EngineActionType]:
        parsed_actions = self.action_parser.parse_actions(actions, state, shared_info)
        returned_actions = {}
        for agent, action in parsed_actions.items():
            del_action = self.delayed_actions[agent]
            ret_action = np.zeros_like(action)
            if len(action) <= self.delay_ticks:
                ret_action[:] = del_action[:len(action)]  # Copy next actions from queue
                del_action[:-len(action)] = del_action[len(action):]  # Shift queue
                del_action[-len(action):] = action  # Add new actions
            else:
                ret_action[:self.delay_ticks] = del_action  # Copy next actions from queue
                ret_action[self.delay_ticks:] = action[:-self.delay_ticks]  # Copy subsequent actions from current
                del_action[:] = action[-self.delay_ticks:]  # Add new actions to queue
            returned_actions[agent] = ret_action
        shared_info["delayed_actions"] = self.delayed_actions
        return returned_actions
