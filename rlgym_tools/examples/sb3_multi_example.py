import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=3,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=VelocityPlayerToBallReward(),  # Simple reward since example code
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, 2)            # Start 2 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=32,                 # PPO calls for multiple epochs
        learning_rate=1e-5,          # Around this is fairly common for PPO
        ent_coef=0.01,               # From PPO Atari
        vf_coef=1.,                  # From PPO Atari
        gamma=gamma,                 # Gamma as calculated using half-life
        verbose=3,                   # Print out all the info as we're going
        batch_size=4096,             # Batch size as high as possible within reason
        n_steps=4096,                # Number of steps to perform before optimizing network
        tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="auto"                # Uses GPU if available
    )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(1_000_000 / env.num_envs), save_path="policy", name_prefix="rl_model")

    model.learn(100_000_000, callback=callback)

    # Now, if one wants to load a trained model from a checkpoint, use this function
    # This will contain all the attributes of the original model
    # Any attribute can be overwritten by using the custom_objects parameter,
    # which includes n_envs (number of agents), which has to be overwritten to use a different amount
    model = PPO.load(
        "policy/rl_model_1000002_steps.zip",
        env,
        custom_objects=dict(n_envs=env.num_envs, _last_obs=None),  # Need this to change number of agents
        device="auto",  # Need to set device again (if using a specific one)
        force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
    )
    # Use reset_num_timesteps=False to keep going with same logger/checkpoints
    model.learn(100_000_000, callback=callback, reset_num_timesteps=False)
