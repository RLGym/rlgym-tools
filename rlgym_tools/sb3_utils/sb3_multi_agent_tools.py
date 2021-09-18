import time
from collections import deque
from typing import List

import gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from typing import Dict, Optional, Union

# This is to allow sequential multi_learn calls
globs = {
"LAST_ALL_OBS" : None,
"LAST_MODEL_MAP" : None,
"OBS_SIZE" : None
} # type: Dict[str, Optional[Union[int, list]]]


# This function is heavily based off the collect_rollouts() method of the sb3 OnPolicyAlgorithm
def multi_collect_rollouts(
        env: VecEnv, models: List[PPO], model_map: list, all_last_obs: list,  n_rollout_steps: int,
        obs_size: int, all_callbacks: List[BaseCallback], learning_mask: List[bool]):
    n_steps = 0
    all_last_episode_restarts = [models[model_map[num]]._last_episode_starts for num in range(len(model_map))]
    for model in models:
        model.rollout_buffer.reset()

    for callback in all_callbacks: callback.on_rollout_start()

    models_length = len(models)
    map_length = len(model_map)
    while n_steps < n_rollout_steps:
        # create indexes to replace later
        all_actions = [0 for _ in range(map_length)]
        all_values = [0 for _ in range(map_length)]
        all_log_probs = [0 for _ in range(map_length)]
        all_clipped_actions = [0 for _ in range(map_length)]

        # disgusting dict and list comprehension to put the same model obs together
        per_model_obs = { model_index:
            np.array([
                all_last_obs[obs_index] for obs_index in range(map_length) if model_map[obs_index] == model_index
            ])
            for model_index in range(models_length) if model_index in model_map
        }

        for model_index in range(models_length):
            # get the actions from the policy
            if model_index in model_map:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(per_model_obs[model_index], models[model_index].device)
                    actions, values, log_probs = models[model_index].policy.forward(obs_tensor)
                actions = actions.cpu().numpy()
                clipped_actions = actions #[0] # it is inside an extra layer for some reason, so take it out
                if isinstance(models[model_index], gym.spaces.Box):
                    clipped_actions = np.clip(
                        actions,
                        models[model_index].action_space.low,
                        models[model_index].action_space.high
                    )

                next_index_start = 0
                # put everything back in terms of the model map
                for i in range(model_map.count(model_index)):
                    next_index = model_map.index(model_index, next_index_start)
                    next_index_start = next_index + 1
                    all_clipped_actions[next_index] = clipped_actions[i]
                    all_actions[next_index] = actions[i]
                    all_values[next_index] = values[i]
                    all_log_probs[next_index] = log_probs[i]
        # flatten the actions, then step the env
        flat_clipped_actions = np.array(all_clipped_actions)
        flat_new_obs, flat_rewards, flat_dones, flat_infos = env.step(flat_clipped_actions)
        # split up the returns from the step
        infos_length = len(flat_infos) // map_length
        all_infos = [flat_infos[x*infos_length:(x+1)*infos_length] for x in range(map_length)]
        all_rewards = [flat_rewards[x] for x in range(map_length)]

        # increment num_timesteps
        for obs_index in range(map_length):
            models[model_map[obs_index]].num_timesteps += 1

        # allow the callbacks to run
        for callback in all_callbacks: callback.update_locals(locals())
        if any(callback.on_step() is False for callback in all_callbacks):
            return False, all_last_obs

        # update the info buffer for each model
        for model_index in range(models_length):
            models[model_index]._update_info_buffer(
                [all_infos[num][0] for num in range(map_length) if model_map[num] == model_index]
            ) # this should put the needed infos for each model in
        n_steps += 1

        # reshape for models with discrete action spaces
        for obs_index in range(map_length):
            if isinstance(models[model_map[obs_index]].action_space, gym.spaces.Discrete):
                all_actions[obs_index] = all_actions[obs_index].reshape(-1,1)

        # add data to the rollout buffers
        for model_index in range(models_length):
            if learning_mask[model_index] and model_index in model_map: # skip learing where not necessary
                models[model_index].rollout_buffer.add( # disgusting list comprehension to send all the info to the buffer
                    np.asarray([all_last_obs[num][0] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_actions[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_rewards[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_last_episode_restarts[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    th.tensor([all_values[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    th.tensor([all_log_probs[num] for num in range(len(model_map)) if model_map[num] == model_index])
                )

        # shuffle variables for next iteration
        new_obs_len = len(flat_new_obs) // map_length
        all_last_obs = [flat_new_obs[obs_index * new_obs_len:(obs_index + 1) * new_obs_len] for obs_index in range(map_length)]
        all_last_episode_restarts = flat_dones

    all_last_values, all_last_dones = [], []
    for obs_index in range(len(model_map)):
        with th.no_grad():
            # compute value for the last timestamp
            # the og code uses new_obs where I have last_obs, so I hope this still works since they should hold the same value
            obs_tensor = obs_as_tensor(all_last_obs[obs_index], models[model_map[obs_index]].device)
            _, values, _ = models[model_map[obs_index]].policy.forward(obs_tensor)
            all_last_values.append(values)

    # compute the returns and advantage for each model
    for model_index in range(len(models)):
        if model_index in model_map:
            models[model_index].rollout_buffer.compute_returns_and_advantage(
                last_values=th.tensor([all_last_values[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                dones=np.asarray([all_last_episode_restarts[num] for num in range(len(model_map)) if model_map[num] == model_index])
            )

    for callback in all_callbacks: callback.on_rollout_end()
    return True, all_last_obs

# This function is heavily based off the learn() method of the sb3 OnPolicyAlgorithm
def multi_learn(
        models: List[PPO],
        total_timesteps: int,
        env,
        num_players: int,
        learning_mask: Optional[List[bool]] = None,
        model_map: Optional[list] = None,
        callbacks: List[MaybeCallback] = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MultiPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
):
    model_map = model_map or [n % len(models) for n in range(num_players)]
    learning_mask = learning_mask or [True for _ in range(len(models))]

    # make sure everything lines up
    if not len(models) == len(callbacks) == len(learning_mask):
        raise ValueError("Length of models, callbacks, and learning_mask must all be equal.")

    iteration = 0
    # this for loop is essentially the setup method, done for each model
    if globs['OBS_SIZE'] is None:
        globs['OBS_SIZE'] = len(env.reset()) // len(model_map)  # calculate the length of the each observation
    all_total_timesteps = []
    for model_index in range(len(models)):
        models[model_index].start_time = time.time()
        if models[model_index].ep_info_buffer is None or reset_num_timesteps:
            models[model_index].ep_info_buffer = deque(maxlen=100)
            models[model_index].ep_success_buffer = deque(maxlen=100)

        if models[model_index].action_noise is not None:
            models[model_index].action_noise.reset()

        if reset_num_timesteps:
            models[model_index].num_timesteps = 0
            models[model_index]._episode_num = 0
            all_total_timesteps.append(total_timesteps)
            models[model_index]._total_timesteps = total_timesteps
        else:
            # make sure training timestamps are ahead of internal counter
            all_total_timesteps.append(total_timesteps + models[model_index].num_timesteps)
            models[model_index]._total_timesteps = total_timesteps + models[model_index].num_timesteps

        # leaving out the environment reset that normally happens here, since that will be done for all at once

        if eval_env is not None and models[model_index].seed is not None:
            eval_env.seed(models[model_index].seed)

        eval_env = models[model_index]._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not models[model_index]._custom_logger:
            models[model_index]._logger = utils.configure_logger(
                models[model_index].verbose,
                models[model_index].tensorboard_log,
                tb_log_name + f'_model{model_index}',
                reset_num_timesteps)

        callbacks[model_index] = models[model_index]._init_callback(
            callbacks[model_index], eval_env, eval_freq, n_eval_episodes, log_path=None)


    for callback in callbacks: callback.on_training_start(locals(), globals())
    if globs["LAST_ALL_OBS"] is None:
        flat_last_obs = env.reset()
        globs["LAST_ALL_OBS"] = [flat_last_obs[x*globs['OBS_SIZE']:(x+1)*globs['OBS_SIZE']] for x in range(num_players)]


    # make sure the n_envs is correct for the models
    for model_index in range(len(models)):
        models[model_index].n_envs = model_map.count(model_index)
        models[model_index].rollout_buffer.n_envs = model_map.count(model_index)

    # I assume the correct thing here is to check each model separately for the while condition
    while all([models[i].num_timesteps < all_total_timesteps[i] for i in range(len(models))]):
        continue_training, globs["LAST_ALL_OBS"] = multi_collect_rollouts(
            env, models, model_map, globs["LAST_ALL_OBS"], min(model.n_steps for model in models), globs["OBS_SIZE"], callbacks, learning_mask
        )

        if continue_training is False:
            break

        iteration += 1
        for model_index in range(len(models)):
            if model_index in model_map:
                models[model_index]._update_current_progress_remaining(models[model_index].num_timesteps, total_timesteps)

        # output to the logger
        for model_index in range(len(models)):
            if log_interval is not None and iteration % log_interval == 0 and learning_mask[model_index] and model_index in model_map:
                fps = int(models[model_index].num_timesteps / (time.time() - models[model_index].start_time))
                models[model_index].logger.record("time/iterations", iteration * model_map.count(model_index), exclude="tensorboard")
                if len(models[model_index].ep_info_buffer) > 0 and len(models[model_index].ep_info_buffer[0]) > 0:
                    models[model_index].logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in models[model_index].ep_info_buffer]))
                    models[model_index].logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in models[model_index].ep_info_buffer]))
                models[model_index].logger.record("time/fps", fps)
                models[model_index].logger.record("time/time_elapsed", int(time.time() - models[model_index].start_time), exclude="tensorboard")
                models[model_index].logger.record("time/total_timesteps", models[model_index].num_timesteps, exclude="tensorboard")
                models[model_index].logger.dump(step=models[model_index].num_timesteps)

        for model_index in range(len(models)):
            if learning_mask[model_index] and model_index in model_map: models[model_index].train()


    for callback in callbacks: callback.on_training_end()


    return models