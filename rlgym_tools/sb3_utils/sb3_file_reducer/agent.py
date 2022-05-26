#
# ONLY AN EXAMPLE
#

from stable_baselines3 import PPO
import pathlib
from discrete_act import DiscreteAction


# This is so there are no optimizer load errors, hacky, but it works
class TestOverrideLoad(PPO):
    def _get_torch_save_params(self):
        state_dicts = ["policy"]

        return state_dicts, []


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = TestOverrideLoad.load(str(_path) + '/example_mdl', device='cpu', custom_objects=custom_objects)
        self.parser = DiscreteAction()

    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]
