from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from test_marl_env import RLLibEnv
from ray.tune import register_env
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction

def make_env():
    import rlgym
    team_size = 1
    self_play = True
    n_agents = team_size*2 if self_play else team_size

    env = rlgym.make(action_parser=LookupAction(),
                     team_size=team_size,
                     spawn_opponents=self_play,
                     use_injector=True)

    return RLLibEnv(env)


if __name__ == "__main__":
    register_env("rlgym", lambda c: make_env())

    cfg = ApexDQNConfig()
    cfg.framework("torch")
    cfg.rollouts(num_rollout_workers=2, )
    cfg.resources(num_gpus=1)
    cfg.environment(env="rlgym")
    cfg.num_atoms = 51

    algo = cfg.build()

    for i in range(100000000):
        result = algo.train()
        print("Epoch: {}\n"
              "Mean Reward: {:7.3f}\n"
              "Mean Ep Len: {:7.2f}\n"
              "Timesteps: {}\n"
              "Time elapsed: {:7.2f}\n"
              .format(result["training_iteration"],
                      result["episode_reward_mean"],
                      result["episode_len_mean"],
                      result["timesteps_total"],
                      result["time_total_s"]))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
