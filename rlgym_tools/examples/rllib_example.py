import os
import time

import numpy as np
import ray
from gym.spaces import Box
from ray import tune
from ray.rllib import Policy, TorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward

from rlgym.utils.obs_builders import AdvancedObs

import rlgym

from rlgym_tools.rllib_utils import RLLibEnv

if __name__ == '__main__':
    ray.init(address='auto', _redis_password='5241590000000000')

    def create_env(env_config):
        return RLLibEnv(rlgym.make(self_play=True,
                                   obs_builder=AdvancedObs(),
                                   reward_fn=VelocityPlayerToBallReward()))

    register_env("RLGym", create_env)

    policy = PPOTorchPolicy, Box(-np.inf, np.inf, (107,)), Box(-1.0, 1.0, (8,)), {}

    ppo_trainer = PPOTrainer(
        env="RLGym",
        config={
            "multiagent": {
                "policies": {"ppo_policy": policy},
                "policy_mapping_fn": (lambda agent_id, **kwargs: "ppo_policy"),
                "policies_to_train": ["ppo_policy"],
            },
            "model": {
                "vf_share_layers": True,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,
            "num_cpus_per_worker": 0,
            "framework": "torch"
        }
    )

    for i in range(1000000):
        print("== Iteration", i, "==")

        # improve the PPO policy
        print("-- PPO --")
        result_ppo = ppo_trainer.train()
        print(pretty_print(result_ppo))
    print("Done training")

    # # This does not work for some reason
    # tune.run(
    #     PPOTrainer,
    #     config={
    #         "env": "RLGym",
    #         "multiagent": {
    #             "policies": {"ppo_policy": policy},
    #             "policy_mapping_fn": (lambda agent_id, **kwargs: "ppo_policy"),
    #             "policies_to_train": ["ppo_policy"],
    #         },
    #         "model": {
    #             "vf_share_layers": True,
    #         },
    #         # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #         "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #         "num_workers": 0,
    #         "num_cpus_per_worker": 0,
    #         "framework": "torch"
    #     }
    # )


