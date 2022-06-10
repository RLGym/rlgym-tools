import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from rlgym_tools.extra_rewards.multi_model_rewards import MultiModelReward
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rlgym_tools.sb3_utils.sb3_multi_agent_tools import multi_learn

if __name__ == '__main__':
    frame_skip = 8
    half_life_seconds = 5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    print(f"fps={fps}, gamma={gamma}")

    # models can only be created/loaded after the env, but the model_map and reward_funcs are used to create the env,
    # so we create all of the lists except for the model list here

    # map of players to model indexes, should be of length = n_envs * players_per_env
    model_map = [0, 0, 1, 2, 3, 3, 2, 0]
    # learning mask is the same size as the models list. True for the model to learn.
    learning_mask = [True, False, True, True]
    # some simple rewards for example purposes. reward_funcs should be in the same order as the list of models.
    reward_funcs = [VelocityPlayerToBallReward(), DefaultReward(), VelocityPlayerToBallReward(), DefaultReward()]


    def get_match():
        return Match(
            team_size=2,  # 2v2 for this example because why not
            tick_skip=frame_skip,
            # use the MultiModelReward to handle the distribution of rewards to each model.
            reward_function=MultiModelReward(model_map, reward_funcs),
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(round(fps * 15)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),
            action_parser=DiscreteAction()
        )


    env = SB3MultipleInstanceEnv(get_match, 2)  # Start 2 instances

    # Hyperparameters presumably better than default; inspired by original PPO paper
    models = [PPO('MlpPolicy', env) for _ in range(4)]
    for _ in range(4):
        model = PPO('MlpPolicy', env)
        models.append(model)

    # This saves to specified folder with a specified name
    # callbacks is a list the same length as the list of models, in the same order.
    callbacks = [CheckpointCallback(round(1_000_000 / env.num_envs), save_path="policy", name_prefix=f"multi_{n}") for n
                 in range(4)]

    # It can be a good idea to call multi-learn multiple times in a loop, modifying model_map in-place (if it is not
    # done in-place, the reward functions will desync) to get extra speed by limiting the number of models training at
    # once. (more separate models training = more calculation time each step)
    multi_learn(
        models=models,  # the list of models that will be used
        total_timesteps=10_000_000,  # total timestamps that will be trained for
        env=env,
        callbacks=callbacks,  # list of callbacks, one for each model in the list of models
        num_players=8,  # team_size * num_instances * 2
        model_map=model_map,  # mapping of models to players.
        learning_mask=learning_mask
    )
