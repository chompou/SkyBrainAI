import numpy as np
import torch

from Environments import GymFactory
import gym
from gym import spaces
from collections import deque


class MultithreadGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, thread_int=1, env_int=2, frame_stack=False, frames_int=1):
        super(MultithreadGym, self).__init__()
        self.env = None
        self.factory = GymFactory.Factory(env_int=env_int, thread_int=thread_int)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 60, 60), dtype=np.uint8)
        self.factory.run()
        self.frame_stack = frame_stack
        if frame_stack:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(frames_int, 3, 60, 60), dtype=np.uint8)
            self.frame_stack = frame_stack
            self.frames_int = frames_int
            self.frame_stack_q = deque(maxlen=frames_int)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.frame_stack:
            self.frame_stack_q.append(torch.tensor(observation))
            observation = torch.stack(tuple(self.frame_stack_q))
        return observation, reward, done, info

    def reset(self):
        local_obs = 0
        if self.env is None:
            local_obs = self.load_env()
        elif not self.env.can_quick_reset():
            self.factory.queue_done(self.env)
            local_obs = self.load_env()
        elif self.env.can_quick_reset():
            local_obs = self.env.reset()
        if self.frame_stack:
            for i in range(self.frames_int):
                self.frame_stack_q.append(torch.tensor(local_obs.copy()))
            local_obs = torch.stack(tuple(self.frame_stack_q))
        return local_obs

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_rgb(self):
        return self.env.save_rgb()

    def close(self):
        self.env.close()
        self.factory.stop()

    def load_env(self):
        if self.env is not None:
            self.factory.queue_done(self.env) # Put old environment in reset-machine

        local_obs, local_env = self.factory.get_ready() # Load freshly resatt enviornment
        self.env = local_env
        return local_obs
