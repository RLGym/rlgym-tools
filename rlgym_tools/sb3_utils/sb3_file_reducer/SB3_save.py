from stable_baselines3 import PPO
from typing import Iterable, Optional, Union, Tuple, List
import io
import pathlib

from stable_baselines3.common.save_util import recursive_getattr, save_to_zip_file


# Example implementation of load hack
class TestOverrideLoad(PPO):
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"]

        return state_dicts, []


# Example implementation of save hack
class TestOverride(PPO):
    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Copy so that we do not overwrite original exclude
        exclude_original = exclude

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()
        # So we don't get dict change errors
        params_to_save_2 = params_to_save.copy()

        if params_to_save is not None:
            for file_name, dict_ in params_to_save.items():
                for param_name in exclude_original:
                    if param_name == file_name:
                        params_to_save_2.pop(file_name)

        save_to_zip_file(path, data=data, params=params_to_save_2, pytorch_variables=pytorch_variables)


if __name__ == "__main__":
    test_class = TestOverride.load("exit_save")
    # Not sure if "optimizer" is needed
    test_class.save("reduced_save", exclude=["policy.optimizer", "optimizer"])
    # Only for checking to make sure the save and load work, not needed
    model = TestOverrideLoad.load("reduced_save")
