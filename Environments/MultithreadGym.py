from Environments import GymFactory

from Environments.SkyRunner import CustomEnv


class MultithreadGym(CustomEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, thread_int=1, env_int=2, frame_stack=False, frames_int=1, use_grayscale=False):
        super(MultithreadGym, self).__init__(frame_stack=frame_stack, frames_int=frames_int, use_grayscale=use_grayscale)

        self.factory = GymFactory.Factory(
            env_int=env_int,
            thread_int=thread_int,
            frame_stack=frame_stack,
            frames_int=frames_int,
            use_grayscale=use_grayscale
        )

        self.factory.run()

    def reset(self):
        if self.env is None:
            self.load_env()
        elif not self.env.can_quick_reset():
            self.factory.queue_done(self.env)
            self.load_env()
        return super().reset()

    def load_env(self):
        if self.env is not None:
            self.factory.queue_done(self.env)  # Put old environment in reset-machine

        local_obs, local_env = self.factory.get_ready()  # Load freshly resatt enviornment
        self.env = local_env
        return local_obs

    def close(self):
        super().close()
        self.factory.stop()
