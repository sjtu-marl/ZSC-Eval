"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import multiprocessing as mp
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from typing import List, Tuple, Union

import cloudpickle
import numpy as np
import psutil

from zsceval.utils.util import tile_images


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)

            if "bool" in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset()
            remote.send((ob))
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "get_max_step":
            remote.send((env.max_steps))
        elif cmd == "anneal_reward_shaping_factor":
            env.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            env.reset_featurize_type(data)
        else:
            raise NotImplementedError


class GuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            # MARK
            p.daemon = False  # could cause zombie process
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            # MARK
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def get_max_step(self):
        for remote in self.remotes:
            remote.send(("get_max_step", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def anneal_reward_shaping_factor(self, steps):
        for remote, step in zip(self.remotes, steps):
            remote.send(("anneal_reward_shaping_factor", step))

    def reset_featurize_type(self, featurize_types):
        for remote, featurize_type in zip(self.remotes, featurize_types):
            remote.send(("reset_featurize_type", featurize_type))


# Add s_ob and available_actions compared to worker
def shareworker(remote, parent_remote, env_fn_wrapper, worker_id: int = None):
    parent_remote.close()
    env = env_fn_wrapper.x()
    if worker_id is not None:
        psutil.Process().cpu_affinity([worker_id])
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "anneal_reward_shaping_factor":
            env.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            env.reset_featurize_type(data)
        elif cmd == "load_policy":
            env.load_policy(data)
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        least_used_cpus = sorted(
            [(c_i, c_percent) for c_i, c_percent in enumerate(psutil.cpu_percent(10, percpu=True))],
            key=lambda x: x[1],
        )
        least_used_cpus = [x[0] for x in least_used_cpus]

        self.ps = [
            Process(
                target=shareworker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                    least_used_cpus[work_id % psutil.cpu_count()],
                ),
            )
            for work_id, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        recveds = [remote.poll() for remote in self.remotes]
        while sum(recveds) == 0:
            recveds = [remote.poll() for remote in self.remotes]
        results = [remote.recv() for remote in self.remotes]

        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            obs,
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return obs, np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def anneal_reward_shaping_factor(self, steps):
        for remote, step in zip(self.remotes, steps):
            remote.send(("anneal_reward_shaping_factor", step))

    def reset_featurize_type(self, featurize_types):
        for remote, featurize_type in zip(self.remotes, featurize_types):
            remote.send(("reset_featurize_type", featurize_type))

    def load_policy(self, load_policy_cfgs):
        for remote, load_policy_cfg in zip(self.remotes, load_policy_cfgs):
            remote.send(("load_policy", load_policy_cfg))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)


# Batch Worker
def dummyvecenvworker(remote, parent_remote, env_fns_wrapper: CloudpickleWrapper, worker_id: int = None):
    parent_remote.close()
    env_fns = env_fns_wrapper.x
    share_dummy_vecenv = ShareDummyVecEnv(env_fns)
    # logger.debug("dummyvecenvworker start")
    if worker_id is not None:
        psutil.Process().cpu_affinity([worker_id])
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = share_dummy_vecenv.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = share_dummy_vecenv.reset()
            remote.send((ob, s_ob, available_actions))
            # logger.debug("dummyvecenvworker reset")
        elif cmd == "reset_task":
            ob = share_dummy_vecenv.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = share_dummy_vecenv.render(mode=data)
                remote.send(fr)
            elif data == "human":
                share_dummy_vecenv.render(mode=data)
        elif cmd == "close":
            share_dummy_vecenv.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (
                    share_dummy_vecenv.observation_space,
                    share_dummy_vecenv.share_observation_space,
                    share_dummy_vecenv.action_space,
                )
            )
        elif cmd == "anneal_reward_shaping_factor":
            share_dummy_vecenv.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            share_dummy_vecenv.reset_featurize_type(data)
        elif cmd == "load_policy":
            share_dummy_vecenv.load_policy(data)
        elif cmd == "update_max_return":
            share_dummy_vecenv.update_max_return(data)
        else:
            raise NotImplementedError


class ShareSubprocDummyBatchVecEnv(ShareVecEnv):
    def __init__(self, env_fns, dummy_batch_size: int = 1, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        assert nenvs % dummy_batch_size == 0
        nbatchs = nenvs // dummy_batch_size
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nbatchs)])
        self.dummy_batch_size = dummy_batch_size
        self.nbatchs = nbatchs

        env_fn_batchs = self._split_batch(env_fns)

        least_used_cpus = sorted(
            [(c_i, c_percent) for c_i, c_percent in enumerate(psutil.cpu_percent(10, percpu=True))],
            key=lambda x: x[1],
        )
        least_used_cpus = [x[0] for x in least_used_cpus]

        self.ps = [
            Process(
                target=dummyvecenvworker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn_batch),
                    least_used_cpus[work_id % psutil.cpu_count()],
                ),
            )
            for work_id, (work_remote, remote, env_fn_batch) in enumerate(
                zip(self.work_remotes, self.remotes, env_fn_batchs)
            )
        ]
        for p_i, p in enumerate(self.ps):
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, share_observation_space, action_space)

    def _split_batch(self, data: List):
        return [data[i : i + self.dummy_batch_size] for i in range(0, len(data), self.dummy_batch_size)]

    def _merge_batch(self, data: List[Union[Tuple, List]]):
        return sum(data[1:], start=data[0])

    def step_async(self, actions):
        action_batchs = self._split_batch(actions)
        for remote, action_batch in zip(self.remotes, action_batchs):
            remote.send(("step", action_batch))
        self.waiting = True

    def step_wait(self):
        result_batchs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        (
            obs_batchs,
            share_obs_batchs,
            rew_batchs,
            done_batchs,
            info_batchs,
            available_actions_batchs,
        ) = zip(*result_batchs)
        return (
            self._merge_batch(obs_batchs),
            np.vstack(share_obs_batchs),
            np.vstack(rew_batchs),
            np.vstack(done_batchs),
            self._merge_batch(info_batchs),
            np.vstack(available_actions_batchs),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        result_batchs = [remote.recv() for remote in self.remotes]
        obs_batchs, share_obs_batchs, available_actions_batchs = zip(*result_batchs)
        return (
            self._merge_batch(obs_batchs),
            np.vstack(share_obs_batchs),
            np.vstack(available_actions_batchs),
        )

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return self._merge_batch([remote.recv() for remote in self.remotes])

    def anneal_reward_shaping_factor(self, steps):
        step_batchs = self._split_batch(steps)
        for remote, step_batch in zip(self.remotes, step_batchs):
            remote.send(("anneal_reward_shaping_factor", step_batch))

    def reset_featurize_type(self, featurize_types):
        featurize_type_batchs = self._split_batch(featurize_types)
        for remote, featurize_type_batch in zip(self.remotes, featurize_type_batchs):
            remote.send(("reset_featurize_type", featurize_type_batch))

    def load_policy(self, load_policy_cfgs):
        load_policy_cfg_batchs = self._split_batch(load_policy_cfgs)
        for remote, load_policy_cfg_batch in zip(self.remotes, load_policy_cfg_batchs):
            remote.send(("load_policy", load_policy_cfg_batch))

    def update_max_return(self, max_returns):
        max_return_batchs = self._split_batch(max_returns)
        for remote, max_return_batch in zip(self.remotes, max_return_batchs):
            remote.send(("update_max_return", max_return_batch))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode: str = "rgb_array"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame_batchs = [remote.recv() for remote in self.remotes]
            return np.stack(frame_batchs)


# No available_actions and s_ob compared to shareworker
def infoworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if "bool" in done.__class__.__name__:
                if done:
                    ob, info = env.reset()
            else:
                if np.all(done):
                    ob, info = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob, info = env.reset()
            remote.send((ob, info))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "get_short_term_goal":
            fr = env.get_short_term_goal(data)
            remote.send(fr)
        else:
            raise NotImplementedError


class InfoSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        # self.envs = [fn() for fn in env_fns]

        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self._mp_ctx = mp.get_context("forkserver")
        self.remotes, self.work_remotes = zip(*[self._mp_ctx.Pipe(duplex=True) for _ in range(nenvs)])

        self.ps = [
            self._mp_ctx.Process(
                target=infoworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), np.stack(infos)

    def get_short_term_goal(self, data):
        for remote, da in zip(self.remotes, data):
            remote.send(("get_short_term_goal", da))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="human"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)


def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
            # Qs: why not reset?
        elif cmd == "reset":
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "get_max_step":
            remote.send((env.max_steps))
        else:
            raise NotImplementedError


class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=choosesimpleworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(("reset", choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_max_step(self):
        for remote in self.remotes:
            remote.send(("get_max_step", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "anneal_reward_shaping_factor":
            env.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            env.reset_featurize_type(data)
        else:
            raise NotImplementedError


def chooseworker_aff(remote, parent_remote, env_fn_wrapper, worker_id: int = None):
    parent_remote.close()
    env = env_fn_wrapper.x()
    if worker_id is not None:
        psutil.Process().cpu_affinity([worker_id])
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "anneal_reward_shaping_factor":
            env.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            env.reset_featurize_type(data)
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        least_used_cpus = sorted(
            [(c_i, c_percent) for c_i, c_percent in enumerate(psutil.cpu_percent(10, percpu=True))],
            key=lambda x: x[1],
        )
        least_used_cpus = [x[0] for x in least_used_cpus]

        self.ps = [
            Process(
                target=chooseworker_aff,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                    least_used_cpus[work_id % psutil.cpu_count()],
                ),
            )
            for work_id, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            obs,
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(("reset", choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return obs, np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def anneal_reward_shaping_factor(self, steps):
        for remote, step in zip(self.remotes, steps):
            remote.send(("anneal_reward_shaping_factor", step))

    def reset_featurize_type(self, featurize_types):
        for remote, featurize_type in zip(self.remotes, featurize_types):
            remote.send(("reset_featurize_type", featurize_type))

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseguardworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseGuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=chooseguardworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = False  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(("reset", choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseinfoworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob, info = env.reset(data)
            remote.send((ob, info))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "get_short_term_goal":
            fr = env.get_short_term_goal(data)
            remote.send(fr)
        elif cmd == "anneal_reward_shaping_factor":
            env.anneal_reward_shaping_factor(data)
        elif cmd == "reset_featurize_type":
            env.reset_featurize_type(data)
        else:
            raise NotImplementedError


class ChooseInfoSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self._mp_ctx = mp.get_context("forkserver")
        self.remotes, self.work_remotes = zip(*[self._mp_ctx.Pipe(duplex=True) for _ in range(nenvs)])

        self.ps = [
            self._mp_ctx.Process(
                target=chooseinfoworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(("reset", choose))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), np.stack(infos)

    def get_short_term_goal(self, data):
        for remote, da in zip(self.remotes, data):
            remote.send(("get_short_term_goal", da))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="human"):
        for remote in self.remotes:
            remote.send(("render", mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_featurize_type(self, featurize_types):
        for remote, featurize_type in zip(self.remotes, featurize_types):
            remote.send(("reset_featurize_type", featurize_type))


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def get_max_step(self):
        return [env.max_steps for env in self.envs]

    def anneal_reward_shaping_factor(self, steps):
        for step, env in zip(steps, self.envs):
            env.anneal_reward_shaping_factor(step)

    def reset_featurize_type(self, featurize_types):
        for featurize_type, env in zip(featurize_types, self.envs):
            env.reset_featurize_type(featurize_type)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human", playeridx=None):
        if mode == "rgb_array":
            if playeridx == None:
                return np.array([env.render(mode=mode) for env in self.envs])
            else:
                return np.array([env.render(mode=mode, playeridx=playeridx) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                if playeridx == None:
                    env.render(mode=mode)
                else:
                    env.render(mode=mode, playeridx=playeridx)
        else:
            raise NotImplementedError


class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(list, zip(*results))

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return (
            obs,
            np.array(share_obs),
            np.array(rews),
            np.array(dones),
            infos,
            np.array(available_actions),
        )

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = zip(*results)
        # logger.debug("DummyVecEnv reset")
        # logger.debug(f"share_obs: {type(share_obs)} {len(share_obs)} {share_obs[0].shape}")
        return obs, np.array(share_obs), np.array(available_actions)

    def reset_task(self):
        results = [env.reset() for env in self.envs]
        return np.stack(results)

    def anneal_reward_shaping_factor(self, steps):
        for env, step in zip(self.envs, steps):
            env.anneal_reward_shaping_factor(step)

    def reset_featurize_type(self, featurize_types):
        for env, featurize_type in zip(self.envs, featurize_types):
            env.reset_featurize_type(featurize_type)

    def load_policy(self, load_policy_cfgs):
        for env, load_policy_cfg in zip(self.envs, load_policy_cfgs):
            env.load_policy(load_policy_cfg)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class InfoDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:
                if done:
                    obs[i], infos[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], infos[i] = self.envs[i].reset()
        self.actions = None

        return obs, rews, dones, infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, infos = map(np.array, zip(*results))
        return obs, infos

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def get_short_term_goal(self, data):
        return [env.get_short_term_goal(d) for d, env in zip(data, self.envs)]


class ChooseDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        self.actions = None
        return (
            obs,
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self, reset_choose):
        results = [env.reset(choose) for (env, choose) in zip(self.envs, reset_choose)]
        obs, share_obs, available_actions = zip(*results)
        return obs, np.stack(share_obs), np.stack(available_actions)

    def anneal_reward_shaping_factor(self, steps):
        for step, env in zip(steps, self.envs):
            env.anneal_reward_shaping_factor(step)

    def reset_featurize_type(self, featurize_types):
        for featurize_type, env in zip(featurize_types, self.envs):
            env.reset_featurize_type(featurize_type)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class ChooseSimpleDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        obs = [env.reset(choose) for (env, choose) in zip(self.envs, reset_choose)]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def get_max_step(self):
        return [env.max_steps for env in self.envs]

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class ChooseInfoDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        results = [env.reset(choose) for (env, choose) in zip(self.envs, reset_choose)]
        obs, infos = map(np.array, zip(*results))
        return obs, infos

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def get_short_term_goal(self, data):
        return [env.get_short_term_goal(d) for d, env in zip(data, self.envs)]

    def ft_get_actions(self, args, mode=""):
        assert mode in [
            "apf",
            "utility",
            "nearest",
            "rrt",
        ], "frontier global mode should be in [apf, utility, nearest, rrt]"
        results = [env.ft_get_actions(args, mode=mode) for env in self.envs]
        actions, goals = map(np.array, zip(*results))
        return actions, goals
