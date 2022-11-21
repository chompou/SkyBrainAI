import argparse

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

import SkyRunner

from stable_baselines3.dqn import DQN, MlpPolicy

from CustomBaselines3.DoubleDQN import DoubleDQN


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


def train(env, eval_env=None, eval_freq=2000):
    """
    Train and save the DQN model, for the cartpole problem
    :param args: (ArgumentParser) the input arguments
    """

    model = DoubleDQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-4,
        learning_starts=750,
        buffer_size=30000,
        exploration_fraction=0.45,
        exploration_final_eps=0.075,
        gamma=0.98,
        target_update_interval=3,
        gradient_steps=-1,
        batch_size=150,
        tau=0.98,
        tensorboard_log="./DQN_steve_tensorboard/",
        use_prioritized_replay=True,
        prioritized_replay_eps=1e-5
    )

    model.save("last_model")

    model.learn(total_timesteps=150000,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                #callback=ImageRecorderCallback(env=env)
                )

