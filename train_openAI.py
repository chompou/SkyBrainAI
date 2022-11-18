import argparse

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

import SkyRunner

from stable_baselines3.dqn import DQN, MlpPolicy

class ImageRecorderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ImageRecorderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self):
        image = self.env.get_rgb()
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("trajectory/image", image, exclude=("stdout", "log", "json", "csv"))
        return True


def train(env):
    """
    Train and save the DQN model, for the cartpole problem
    :param args: (ArgumentParser) the input arguments
    """

    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-2,
        learning_starts=750,
        buffer_size=50000,
        exploration_fraction=0.55,
        exploration_final_eps=0.075,
        gamma=0.975,
        tau=0.85,
        tensorboard_log="./DQN_steve_tensorboard/"
    )
    model.learn(total_timesteps=10000, callback=ImageRecorderCallback(env=env))

    obs = env.reset()
    r = 0
    for i in range(5000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()

        if done:
            print(r)
            r = 0
            obs = env.reset()

    print("Saving model to cartpole_model.zip")
    model.save("cartpole_model.zip")
