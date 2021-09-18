import numpy as np
from rlgym.envs import Match
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from rlgym_tools.extra_rewards.multi_model_rewards import MultiModelReward
from rlgym_tools.sb3_utils import SB3MultiDiscreteWrapper, SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan, VecMonitor

from rlgym_tools.sb3_utils.sb3_multi_agent_tools import multi_learn


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

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

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=2,  # 2v2 for this example because why not
            tick_skip=frame_skip,
            # use the MultiModelReward to handle the distribution of rewards to each model.
            reward_function=MultiModelReward(model_map, reward_funcs),
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 15)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState()
        )

    env = SB3MultipleInstanceEnv(get_match, 2)     # Start 2 instances
    env = SB3MultiDiscreteWrapper(env)                      # Convert action space to multidiscrete
    env = VecCheckNan(env)                                  # Optional
    env = VecMonitor(env)                                   # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    models = []
    for _ in range(4):
        model = PPO(
            'MlpPolicy',
            env,
            n_epochs=32,                 # PPO calls for multiple epochs, SB3 does early stopping to maintain target kl
            learning_rate=1e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=4096,          # Batch size as high as possible within reason
            n_steps=4096,             # Number of steps to perform before optimizing network
            tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )
        models.append(model)

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    # callbacks is a list the same length as the list of models, in the same order.
    callbacks = [CheckpointCallback(round(1_000_000 / env.num_envs), save_path="policy", name_prefix=f"multi_{n}") for n in range(4)]

    # It can be a good idea to call multi-learn multiple times in a loop, modifying model_map in-place (if it is not
    # done in-place, the reward functions will desync) to get extra speed by limiting the number of models training at
    # once. (more separate models training = more calculation time each step)
    multi_learn(
        models= models, # the list of models that will be used
        total_timesteps= 10_000_000, # total timestamps that will be trained for
        env= env,
        callbacks= callbacks, # list of callbacks, one for each model in the list of models
        num_players= 8, # team_size * num_instances
        model_map= model_map, # mapping of models to players.
        learning_mask= learning_mask
    )

    exit(0)

    # Now, if one wants to load a trained model from a checkpoint, use this function
    # This will contain all the attributes of the original model
    # Any attribute can be overwritten by using the custom_objects parameter,
    # which includes n_envs (number of agents), which has to be overwritten to use a different amount
    model = PPO.load("policy/rl_model_1000002_steps.zip", env, custom_objects=dict(n_envs=1))
    # Use reset_num_timesteps=False to keep going with same logger/checkpoints













