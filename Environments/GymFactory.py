import threading
from Environments import SkyRunner
from threading import Thread
import queue
import time

exit_flag = 0
q = queue.Queue()
r = queue.Queue()
q_lock = threading.Lock()
r_lock = threading.Lock()


class Factory:
    def __init__(self, env_int=5, thread_int=5):
        super().__init__()
        global q
        global r
        global q_lock
        global r_lock
        self.threads = []
        for i in range(env_int):
            env = SkyRunner.CustomEnv()
            q.put(env)
        for j in range(thread_int):
            self.threads.append(Worker(j, "Reloader"))

    def get_ready(self):
        global r
        global r_lock
        r_lock.acquire()
        obs, env = r.get()
        r_lock.release()
        return obs, env

    def queue_done(self, done_env):
        global q
        global q_lock
        q_lock.acquire()
        q.put(done_env)
        q_lock.release()

    def run(self):
        for t in self.threads:
            t.start()

    def stop(self):
        global exit_flag
        global q_lock
        global r_lock
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
    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        global q
        global r
        global q_lock
        global r_lock
        self.thread_id = thread_id
        self.name = name

    def run(self) -> None:
        print("starting", self.name, self.thread_id)
        self.reload()
        print("stopping", self.name, self.thread_id)

    def reload(self):
        global q
        global r
        global q_lock
        global r_lock
        global exit_flag
        while not exit_flag:
            q_lock.acquire()
            if not q.empty():
                local_env = q.get()
                print(self.thread_id, "got", local_env)
                q_lock.release()
                obs = local_env.reset()
                if isinstance(obs, int):
                    local_env.__exit__()
                    local_env = SkyRunner.CustomEnv()
                    obs = local_env.reset()
                r_lock.acquire()
                r.put((obs, local_env))
                r_lock.release()
            else:
                q_lock.release()
                time.sleep(1)
