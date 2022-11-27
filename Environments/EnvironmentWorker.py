import threading
import traceback

from Environments import SkyRunner
from threading import Thread
import queue

exit_flag = 0
q = queue.Queue()
r = queue.Queue()
q_lock = threading.Lock()
r_lock = threading.Lock()

"""
Class which handles threads for loading and resetting SkyRunnerEnvironmentInstances.
"""
class Producer:
    def __init__(self, env_int=5, thread_int=5, frame_stack=False, frames_int=1, use_grayscale=False):
        super().__init__()

        self.frame_stack = frame_stack
        self.frames_int = frames_int
        self.use_grayscale = use_grayscale

        global q
        global r
        global q_lock
        global exit_flag

        exit_flag = 0

        self.threads = []
        for i in range(env_int):
            env = SkyRunner.build_mission(
                use_grayscale=self.use_grayscale)
            q.put(env)

        for j in range(thread_int):
            self.threads.append(Worker(j, "Reloader",
                                       use_grayscale=self.use_grayscale))

    """
    Gets a Environment from the ready-queue, will block if no items are ready yet
    """

    def get_ready(self):
        global r
        global r_lock
        print("Waiting for ready enviornment")
        r_lock.acquire()
        obs, env = r.get(block=True)
        r_lock.release()
        print("Ready enviornment received")
        return obs, env

    """
    Put an Environment into the queue for environments which needs reset. The first worker to be ready will pick this
    up and begin a reset.
    """

    def queue_done(self, done_env):
        global q
        q.put(done_env)

    def run(self):
        for t in self.threads:
            t.start()

    def stop(self):
        global exit_flag
        global q
        global r
        exit_flag = 1
        for t in self.threads:
            t.join()
        for e in range(q.qsize()):
            q.get().close()
        for et in range(r.qsize()):
            r.get()[1].close()


class Worker(Thread):
    def __init__(self, thread_id, name, use_grayscale):
        threading.Thread.__init__(self)
        global q
        global r
        global q_lock
        global r_lock
        self.thread_id = thread_id
        self.name = name
        self.use_grayscale = use_grayscale

    def run(self) -> None:
        print("starting", self.name, self.thread_id)
        try:
            self.reload()
        except:
            print("Error in thread %d. " % self.thread_id)
            print(traceback.format_exc())
        print("stopping", self.name, self.thread_id)

    def reload(self):
        global q
        global r
        global exit_flag
        while not exit_flag:
            try:
                q_lock.acquire()
                # Attempt to get environment from queue for environments which needs reset
                local_env = q.get(block=True, timeout=5)
                print(
                    "ThreadID: %d has received an enviornment from queue. Reset of environement is being prepeared" % (
                        self.thread_id))
                q.task_done()
                q_lock.release()

                # Environment received: Begin reset.
                obs = local_env.reset()

                # If observation is an integer, then the reset has failed. Try again.
                while isinstance(obs, int):
                    print("Failed to load/reset environment, retrying... ThreadID: %d" % self.thread_id)
                    local_env = SkyRunner.build_mission(use_grayscale=self.use_grayscale)
                    obs = local_env.reset()

                r.put((obs, local_env), block=False)
                print("ThreadID %d put complete environment into ready-queue." % self.thread_id)
            except queue.Empty:
                q_lock.release()
                # print("Queue-GET timed out. Trying again, if not exit_flag has been sat. ThreadID: %d" % self.thread_id)
                continue
