import multiprocessing as mp
import os
import time
from typing import Optional, List, Union, Any

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
from stable_baselines3.common.vec_env.subproc_vec_env import _worker

from rlgym.envs import Match
from rlgym.gym import Gym


class SB3MultipleInstanceEnv(SubprocVecEnv):
    """
    Class for launching several Rocket League instances into a single SubprocVecEnv for use with Stable Baselines.
    """
    MEM_INSTANCE_LAUNCH = 3.5e9
    MEM_INSTANCE_LIM = 4e6

    def __init__(self, path_to_epic_rl, num_instances, match_args_func, wait_time=60):
        """
        :param path_to_epic_rl: Path to the Rocket League executable of the Epic version.
        :param num_instances: the number of Rocket League instances to start up.
        :param match_args_func: a function which produces the arguments for the Match object.
                                Needs to be a function so that each subprocess can call it and get their own objects.
        :param wait_time: the time to wait between launching each instance. Default one minute.
        """
        auto_instances = num_instances == "auto"
        if auto_instances:
            import psutil
            vm = psutil.virtual_memory()
            # Need 3.5GB to launch, reduces to 350MB after a while
            est_proc_mem = round((vm.available - self.MEM_INSTANCE_LAUNCH) / self.MEM_INSTANCE_LAUNCH)
            est_proc_cpu = os.cpu_count()
            est_proc = min(est_proc_mem, est_proc_cpu)
            print(f"Estimated supported processes:", est_proc)
            num_instances = est_proc

        def spawn_process():
            match = Match(**match_args_func())
            env = Gym(match, pipe_id=os.getpid(), path_to_rl=path_to_epic_rl, use_injector=True)
            # env.reset()
            # while True:
            #     env.step(np.zeros((match.agents, 8)))
            return env

        match_args = match_args_func()

        # super().__init__([])  Super init intentionally left out since we need to launch processes with delay

        env_fns = [spawn_process for _ in range(num_instances)]

        # START - Code from SubprocVecEnv class
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # Fork is not a thread safe method (see issue #217)
        # but is more user friendly (does not require to wrap the code in
        # a `if __name__ == "__main__":`)
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):

            # if auto_instances:  # ADDED - Waiting for memory usage to go down
            #     while psutil.virtual_memory().free < self.MEM_INSTANCE_LAUNCH:  # noqa
            #         print("Memory usage too high, waiting 10 minutes...")
            #         time.sleep(10 * 60)

            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

            time.sleep(wait_time)  # ADDED - Waits between starting Rocket League instances

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        # END - Code from SubprocVecEnv class

        self.n_agents_per_env = match_args.get("team_size") * (1 + match_args.get("self_play", False))
        self.num_envs = num_instances * self.n_agents_per_env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = sum((remote.recv() for remote in self.remotes), [])
        return np.asarray(obs)

    def step_async(self, actions: np.ndarray) -> None:
        i = 0
        for remote in self.remotes:
            remote.send(("step", actions[i: i + self.n_agents_per_env, :]))
            i += self.n_agents_per_env
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        obs = sum(obs, [])
        rews = sum(rews, [])
        dones = [d for d in dones for _ in range(self.n_agents_per_env)]
        infos = [i for i in infos for _ in range(self.n_agents_per_env)]
        return np.asarray(obs), np.array(rews), np.array(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        res = super(SB3MultipleInstanceEnv, self).seed(seed)
        return [r for r in res for _ in range(self.n_agents_per_env)]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        # Override to prevent out of bounds
        indices = self._get_indices(indices)
        return [self.remotes[i // self.n_agents_per_env] for i in indices]