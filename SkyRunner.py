import minedojo
import numpy as np

import Skyrunner_mission_interpreter
import gym
from gym import spaces

image_size = (660, 600)


def create_env():
    env = minedojo.make(
        "open-ended",
        image_size=image_size,
        generate_world_type='flat',
        flat_world_seed_string="0",
        start_position=dict(x=0, y=2, z=0, yaw=0, pitch=0),
        fast_reset=True,
        start_time=6000,
        allow_time_passage=False,
        drawing_str="""
        <DrawCuboid x1="0" y1="0" z1="0" x2="-3" y2="0" z2="-5" type="dirt"/>
        <DrawCuboid x1="0" y1="-1" z1="0" x2="-3" y2="-1" z2="-5" type="dirt"/>
        <DrawCuboid x1="0" y1="1" z1="0" x2="-3" y2="1" z2="-5" type="grass"/>
        <DrawCuboid x1="1" y1="0" z1="-3" x2="3" y2="0" z2="-5" type="dirt"/>
        <DrawCuboid x1="1" y1="-1" z1="-3" x2="3" y2="-1" z2="-5" type="dirt"/>
        <DrawCuboid x1="1" y1="1" z1="-3" x2="3" y2="1" z2="-5" type="grass"/>
        <DrawCuboid x1="3" y1="2" z1="-5" x2="3" y2="6" z2="-5" type="log"/>
        """,
        use_lidar=True,
        lidar_rays=[(0, 0, 999)]
    )

    return Skyrunner_mission_interpreter.Mission(survival=True, explore=True, episode_length=100, min_y=-3,
                                                 env=env)


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, ):
        super(CustomEnv, self).__init__()
        self.env = None
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 60, 60), dtype=np.uint8)

    def step(self, action):
        observation, reward, done, info = self.env.stepNum(action)
        return observation, reward, done, info

    def reset(self):
        if self.env is None:
            self.env = create_env()
        for i in range(10):
            try:
                return self.env.reset()
            except:
                self.env = None
        return -1

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_rgb(self):
        return self.env.save_rgb()

    def close(self):
        self.env.quit()
        self.__exit__()
