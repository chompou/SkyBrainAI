import numpy as np
import torch
from Environments import GymFactory, SkyRunner
import gym
from gym import spaces
from collections import deque
from typing import Callable


class MultithreadGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_frames: int, preprocess: Callable, thread_int = 5, env_int=5):
        super(MultithreadGym, self).__init__()
        self.env = None
        self.factory = GymFactory.Factory(env_int=env_int, thread_int=thread_int)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 60, 60), dtype=np.uint8)
        self.n_frames = n_frames
        self.factory.run()

    def step(self, action, original: bool = False):
        original_frames = []
        frame, observation, reward, done, info = self.env.step(action)
        if original:
            original_frames.append(frame)
            
        if self._preprocess is not None:
            frame_ = self._preprocess(self._world, self._stage, frame)
        else:
            frame_ = frame
        self.frames.appendleft(frame_)
        if not original:
            return torch.stack(tuple(self.frames)), observation, reward, done, info
        else:
            return [torch.stack(tuple(self.frames)), list(original_frames)], observation, reward, done, info

    def reset(self, original=False):
        ready = False
        if self.env is not None:
            self.factory.queue_done(self.env)
        while not ready:
            local_obs, local_env = self.factory.get_ready()
            if isinstance(local_obs, int):
                env = SkyRunner.CustomEnv()
                self.factory.queue_done(env)
            else:
                ready = True
        self.env = local_env
        frame = self._env.reset()
        frame_ = self._preprocess(self._world, self._stage, frame)
        self.frames = deque([frame_]*self.n_frames, self.n_frames)
        if not original:
            return torch.stack(tuple(self.frames))
        else:
            return torch.stack(tuple(self.frames)), frame

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_rgb(self):
        return self.env.save_rgb()

    def close(self):
        self.factory.stop()
        self.__exit__()
