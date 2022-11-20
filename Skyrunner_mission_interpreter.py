import random
import numpy as np
import torch
import torchvision
from PIL import Image
from datetime import datetime

from torchvision.transforms import InterpolationMode


def spawn():
    x_z = [
        (0, 0), (-1, 0), (-2, 0), (-3, 0),
        (0, -1), (-1, -1), (-2, -1), (-3, -1),
        (0, -2), (-1, -2), (-2, -2), (-3, -2),
        (0, -3), (-1, -3), (-2, -3), (-3, -3),
        (0, -4), (-1, -4), (-2, -4), (-3, -4),
        (0, -5), (-1, -5), (-2, -5), (-3, -5),
        (1, -3), (1, -4), (1, -5),
        (1, -3), (2, -3), (3, -3),
    ]
    rad = [-90, 0, 90, 180]

    pos = random.choice(x_z)
    yaw = random.choice(rad)
    return pos[0], 2, pos[1], yaw, 0


class Mission:
    def __init__(self, explore=False, obs_simplify=True, mine=False, survival=False, collect_amount=None,
                 per_item_reward=None, episode_length=1000, env=None, prev_distance=0, min_y=0):
        self.attack = None
        if per_item_reward is None:
            per_item_reward = []
        if collect_amount is None:
            collect_amount = []

        self.resize = torchvision.transforms.Resize((60, 60), interpolation=InterpolationMode.NEAREST, antialias=None, )
        self.obs_simplify = obs_simplify
        self.explore = explore
        self.mine = mine
        self.survival = survival
        self.per_item_reward = per_item_reward
        self.collect_amount = collect_amount
        self.delta = ({}, {}, {}, {'distance_travelled_cm': prev_distance, 'rays': ['init', 1000]})
        self.episode = 0
        self.episode_length = episode_length
        self.env = env
        self.min_y = min_y

    def translate_action(self, action):
        forward = action == 0
        backward = action == 1
        # left = action == 2
        # right = action == 3
        cam_pitch_l = action == 2
        cam_pitch_r = action == 3
        cam_yaw_up = action == 4
        cam_yaw_down = action == 5
        self.attack = action == 6

        return [
            1 if forward else 2 if backward else 0,  # 0
            0,  # if left else 2 if right else 0,  # 1
            0,  # 2
            11 if cam_yaw_up else 13 if cam_yaw_down else 12,  # 3
            11 if cam_pitch_l else 13 if cam_pitch_r else 12,  # 4
            3 if self.attack and self.delta[3].get('rays')[0] == 'wood' and self.delta[3].get('rays')[1] < 3 else 0,
            # 5
            0,  # 6
            0,  # 7
        ]

    def eval(self, obs, reward, done, info):
        if reward is None:
            reward = 0
        self.episode += 1
        if self.min_y:
            if info.get('ypos') < self.min_y:
                done = True
                reward -= 500
        if self.obs_simplify:
            obs = self.rgb_simplify(obs.get('rgb'))
        if self.episode >= self.episode_length:
            done = True
        if self.survival:
            if info.get('is_dead'):
                done = True
            if info.get('living_death_event_fired'):
                done = True
        if self.attack:
            if info.get('rays')[0] == 'wood' and info.get('rays')[1] < 2:
                reward += 1000
                done = True
        if self.explore:
            new = info.get('distance_travelled_cm') if info.get('distance_travelled_cm') is not None else 0
            old = self.delta[3].get('distance_travelled_cm') if self.delta[3].get(
                'distance_travelled_cm') is not None else 0
            reward += new - old
        self.attack = None
        self.delta = (obs, reward, done, info)
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        self.__init__(explore=self.explore, obs_simplify=self.obs_simplify, mine=self.mine, survival=self.survival,
                      collect_amount=self.collect_amount,
                      per_item_reward=self.per_item_reward, episode_length=self.episode_length, env=self.env,
                      prev_distance=self.delta[3].get('distance_travelled_cm') if self.delta[3].get(
                          'distance_travelled_cm') is not None else 0, min_y=self.min_y)

        x, y, z, yaw, pitch = spawn()

        self.env.teleport_agent(x=x, y=y, z=z, yaw=yaw, pitch=pitch)

        for i in range(2):
            _, __, ___, ____ = self.stepNum(100)

        return _

    def step(self, action):
        _, __, ___, ____ = self.env.step(action)
        return self.eval(_, __, ___, ____)

    def quit(self):
        self.env.__exit__()

    def stepNum(self, num):
        action = self.translate_action(num)
        return self.step(action)

    def render(self):
        self.env.render()

    def save_rgb(self, array):
        now = datetime.now()
        src = 'renders/' + str(now) + 'render.png'
        n1 = array[0, :, :]
        n2 = array[1, :, :]
        n3 = array[2, :, :]

        result = np.dstack((n1, n2, n3))
        result = Image.fromarray(result.astype(np.uint8))
        result.save(src)

    def test_teleport(self):
        x, y, z, yaw, pitch = spawn()
        self.env.teleport_agent(x=x, y=y, z=z, yaw=yaw, pitch=pitch)
        a, _, __, ___ = self.stepNum(100)
        return a

    def rgb_simplify(self, array):
        return self.resize(torch.Tensor(array[:, :600, :600].copy())).numpy().astype(dtype='uint8')
