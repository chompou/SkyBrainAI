import numpy as np
from Environments import GymFactory, SkyRunner
import gym
from gym import spaces


class MultithreadGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, thread_int=5, env_int=5):
        super(MultithreadGym, self).__init__()
        self.env = None
        self.factory = GymFactory.Factory(env_int=env_int, thread_int=thread_int)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 60, 60), dtype=np.uint8)
        self.factory.run()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
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
