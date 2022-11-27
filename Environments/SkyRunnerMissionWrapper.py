import random
import numpy as np
import torch
import torchvision
from stable_baselines3.common.logger import Image

from torchvision.transforms import InterpolationMode


class EnvironmentMissionWrapper:
    """
    Wrapper to add custom reward-scheme, action-space/translator and to transform observations.
    """

    def __init__(self, explore=False,
                 obs_simplify=True,
                 obs_grayscale=True,
                 survival=False,
                 collect_amount=None,
                 spawn_locations=None,
                 episode_length=1000,
                 env=None,
                 min_y=0
                 ):

        self.attack = None

        self.resize = torchvision.transforms.Resize((60, 60), interpolation=InterpolationMode.NEAREST,
                                                    antialias=None, )  # For simplicying the observation-space
        self.grayscale = torchvision.transforms.Grayscale()  # For applying converting observation to grayscale if requested.

        self.obs_simplify = obs_simplify  # Wether or not to downsize observation
        self.obs_grayscale = obs_grayscale  # Should the observation be grayscaled?
        self.explore = explore  # Should the agent recevie reward for exploring/walking
        self.survival = survival  # Should survival mode be activated
        self.delta = None  # Initiated in init_env
        self.episode = 0  # Episode tracker
        self.episode_length = episode_length  # Max timestamps of any given episode
        self.env = env  # Underlyiung enviornment
        self.min_y = min_y  # Minimum allowed y-value, death will occur if the agent is lower than this
        self.spawn_locations = spawn_locations  # Allowed spawn locations, this will be multi dimensional when more than one Island is used.
        self.location_index = -1  # Index to keep track of which Island the agent is currently on
        self.max_location_index = 0  # Max amount of Islands the agent may traverse
        self.chopped_dist = 5  # Distanse the agent is allowed to break blocks
        self.chopped = 0  # Number of blocks which are destroyed on the current episode
        self.previous_episode_move = 0  # Previous episode which the agent moved
        self.wood_count = 0  # Count of wood-blocks picked up by the agent

        self.REWARD_DEATH_PUNISHMENT = -5  # Penalty for falling off the skyblock
        self.REWARD_PICK_UP_WOOD = 500  # Reward for picking up wood-item from ground
        self.REWARD_HIT_ON_WOOD = 3  # Reward for hitting wood-block
        self.REWARD_BONUS_AT_LEVEL_COMPLETE = 500  # Reward for completing the whole level
        self.MISSION_COMPLETE_WOOD_COUNT = 5  # Number of wood-blocks to pick up before level is solved

        if spawn_locations is not None:
            self.max_location_index = len(spawn_locations)

        self.init_env()

    def init_env(self):
        self.delta = ({}, {}, {}, {'distance_travelled_cm': 0, 'rays': ['init', 1000]})
        self.episode = 0
        self.chopped = 0
        self.previous_episode_move = 0
        self.wood_count = 0

    def translate_action(self, action):
        """
        Transtale a discrete action into an Box-action.
        """
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
            3 if self.attack and (
                    self.delta[3].get('rays')[0] == 'wood' or self.delta[3].get('rays')[0] == 'leaves') and
                 self.delta[3].get('rays')[
                     1] < self.chopped_dist else 0,
            # 5
            0,  # 6
            0,  # 7
        ]

    def eval(self, obs, reward, done, info):
        """
        Evaluate data directly from a underlying minedojo instance. This evaluation performs mission-specific evaluation
        and corrects the reward, done-variable and verifies if an attack is allowed.

        The new environment-data is returned
        """
        if reward is None:
            reward = 0

        self.episode += 1

        if self.episode >= self.episode_length:  # Cut episode if it is longer than allowed
            done = True
        if self.min_y:  # Check if agent has fallen below min-y. Terminate episode if so.
            if info.get('ypos') < self.min_y:
                done = True
                reward += self.REWARD_DEATH_PUNISHMENT
        if self.survival:  # ... or terminate if agent is dead
            if info.get('is_dead'):
                done = True
            if info.get('living_death_event_fired'):
                done = True

        wood_sum = self.get_wood_count(obs)  # Number of wood-blocks collected by Steve
        if wood_sum > self.wood_count:  # If there has been new pickups
            wood_new = wood_sum - self.wood_count
            reward += int((wood_new * self.REWARD_PICK_UP_WOOD))  # Give appropriate reward for new pickups
            self.wood_count = wood_sum

        if self.wood_count == self.MISSION_COMPLETE_WOOD_COUNT:  # Mission completed!! Wee :D
            done = True
            reward += self.REWARD_BONUS_AT_LEVEL_COMPLETE

        if self.attack:  # If agent has attempted to attack, verify if any blocks are broken.
            self.attack = None
            curr_b = info.get('rays')[0]
            curr_d = info.get('rays')[1]
            prev_b = self.delta[3].get('rays')[0]
            prev_d = self.delta[3].get('rays')[1]
            if curr_b == 'wood' and curr_d < self.chopped_dist:
                reward += self.REWARD_HIT_ON_WOOD
            if (prev_b == 'wood' or prev_b == 'leaves') and prev_d < self.chopped_dist and (
                    prev_d != curr_d or prev_b != curr_b):
                self.chopped = 1

        if self.explore:  # Reward for walking
            new = info.get('distance_travelled_cm') if info.get('distance_travelled_cm') is not None else 0
            old = self.delta[3].get('distance_travelled_cm') if self.delta[3].get(
                'distance_travelled_cm') is not None else 0
            d_pos = new - old  # Delta position: position moved since last time

            if d_pos:  # If moved more than 0
                reward += (d_pos / 100.0)  # Divide by 100 to convert cm -> m.
                self.previous_episode_move = self.episode
            elif (self.episode - self.previous_episode_move) > 150:
                done = True

        if self.obs_simplify:  # Downscale observation before returning it
            obs = self.rgb_simplify(obs.get('rgb'))

        self.delta = (obs, reward, done, info)
        return obs, reward, done, info

    def reset(self):
        if self.chopped:
            self.location_index += 1
            print("Broken block detected. Moving to location %d" % self.location_index)

        if not self.can_quick_reset() or self.location_index == -1:
            self.env.reset()
            self.location_index = 0
        else:
            self.quick_reset()

        self.init_env()

        x, y, z, yaw, pitch = self.spawn()

        self.env.teleport_agent(x=x, y=y, z=z, yaw=yaw, pitch=pitch)

        ## Let Steve hit the ground
        for i in range(2):
            obs, reward, done, info = self.step(100)  ## Perform 2 no-ops

        return obs

    def actStep(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.eval(obs, reward, done, info)

    def quit(self):
        self.env.__exit__()

    def step(self, num):
        action = self.translate_action(num)
        return self.actStep(action)

    def render(self, mode):
        self.env.render(mode)

    def save_rgb(self):
        array = self.delta[0]

        result = np.dstack(array)
        return Image(result, "HWC")

    def rgb_simplify(self, array):
        """
        Crop, downscale and, if requestedm grayscale the observation
        """
        array = torch.Tensor(array.copy())
        if self.obs_grayscale:
            array = self.grayscale(array)
        return self.resize(array[:, :600, :600]).numpy().astype(dtype='uint8')

    def spawn(self):
        """
        Spawn in random position of the Island.
        """
        if self.spawn_locations is None:
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
        else:
            x_z = self.spawn_locations[self.location_index]

        rad = [-90, 0, 90, 180]

        pos = random.choice(x_z)
        yaw = random.choice(rad)
        return pos[0], 2, pos[1], yaw, 0

    def can_quick_reset(self):
        """
        If there are still unused Islands available on the current map, the agent may quick-reset.
        """
        return self.location_index < self.max_location_index

    def quick_reset(self):
        """
        Kill, clear inventory, and respawn the agent to a new Island.
        """

        self.env.kill_agent()
        attempts = 5

        # Clear inventory sporadically fails, attempting again 5 times if this happends.
        for i in range(attempts):
            try:
                self.env.clear_inventory()
                self.env.execute_cmd("/weather clear")

                print("Inventory and weather cleared!")
                break
            except:
                print("Unable to clear inventory or weather... Retyring %d more times" % (attempts - i))

    def get_wood_count(self, obs):

        """
        Count number of wood-blocks the agent has in his inventory
        """

        inventory_name = obs.get('inventory').get('name')
        wood_index = np.where(inventory_name == 'log')
        wood_sum = 0
        for i in range(len(wood_index[0])):
            wood_sum += obs.get('inventory').get('quantity')[wood_index[i]][0]
        return wood_sum
