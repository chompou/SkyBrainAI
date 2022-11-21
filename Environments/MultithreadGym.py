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
        return local_obs

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_rgb(self):
        return self.env.save_rgb()

    def close(self):
        self.factory.stop()
        self.__exit__()