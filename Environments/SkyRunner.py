import traceback

import minedojo
import numpy as np
from Environments import Skyrunner_mission_interpreter
from environment_drawer import draw_skyblock_grid
import gym
from gym import spaces

image_size = (660, 600)


def create_env():
    draw_string, spawn_locations = draw_skyblock_grid(100, 100, 50)

    env = minedojo.make(
        "open-ended",
        image_size=image_size,
        generate_world_type='flat',
        flat_world_seed_string="0",
        start_position=dict(x=0, y=2, z=0, yaw=0, pitch=0),
        # fast_reset=True,
        start_time=6000,
        allow_time_passage=False,
        drawing_str=draw_string,
        use_lidar=True,
        allow_mob_spawn=False,
        break_speed_multiplier=10,
        lidar_rays=[(0, 0, 999)]
    )

    return Skyrunner_mission_interpreter.Mission(survival=True,
                                                 explore=True,
                                                 episode_length=1000,
                                                 min_y=-3,
                                                 env=env,
                                                 spawn_locations=spawn_locations
                                                 )


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

        try:
            return self.env.reset()
        except:
            print("Unable to reset enviornment due to an exception")
            print(traceback.format_exc())
            try:
                self.env.quit()
            except:
                print("Failed to terminate broken enviornment.")
            self.env = None

            return -1

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_rgb(self):
        return self.env.save_rgb()

    def close(self):
        if self.env is not None:
            self.env.quit()

    def can_quick_reset(self):
        return self.env.can_quick_reset()
