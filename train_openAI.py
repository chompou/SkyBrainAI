from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.dqn import MlpPolicy

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


def train(env,
          eval_env,
          total_timesteps=150000,
          eval_freq=100,
          n_eval_episodes=5,
          learning_rate=1e-3,
          learning_starts=10000,
          buffer_size=500000,
          batch_size=50,
          exploration_initial_eps=1.0,
          exploration_fraction=0.65,
          exploration_final_eps=0.005,
          gamma=0.98,
          target_update_interval=1,
          gradient_steps=-1,
          tau=0.96,
          use_prioritized_replay=True,
          prioritized_replay_eps=1e-5,
          prioritized_replay_initial_beta=1.0,
          prioritized_replay_final_beta=0.1,
          prioritized_replay_beta_fraction=0.55,
):
    """
    Train and save the DQN model, for the cartpole problem
    :param args: (ArgumentParser) the input arguments
    """

    model = DoubleDQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        exploration_initial_eps=exploration_initial_eps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        gamma=gamma,
        target_update_interval=target_update_interval,
        gradient_steps=gradient_steps,
        batch_size=batch_size,
        tau=tau,
        tensorboard_log="./DQN_steve_tensorboard/",
        use_prioritized_replay=use_prioritized_replay,
        prioritized_replay_eps=prioritized_replay_eps,
        prioritized_replay_initial_beta=prioritized_replay_initial_beta,
        prioritized_replay_beta_fraction =prioritized_replay_beta_fraction,
        prioritized_replay_final_beta=prioritized_replay_final_beta
    )

    model.save("last_model")

    model.learn(total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                )

