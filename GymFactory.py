import threading
import SkyRunner
from threading import Thread
import queue
import time

exit_flag = 0


class Factory:
    def __init__(self, env_int=5, thread_int=5):
        super().__init__()
        self.q = queue.Queue()
        self.q_lock = threading.Lock()
        self.r = queue.Queue()
        self.r_lock = threading.Lock()
        self.threads = []
        for i in range(env_int):
            env = SkyRunner.CustomEnv()
            self.q.put(env)
        for j in range(thread_int):
            self.threads.append(Worker(j, "Reloader", self.q, self.r, self.q_lock, self.r_lock))

    def get_ready(self):
        self.r_lock.acquire()
        obs, env = self.r.get()
        self.r_lock.release()
        return obs, env

    def queue_done(self, done_env):
        self.q.put(done_env)

    def run(self):
        for t in self.threads:
            t.start()

    def stop(self):
        global exit_flag
        exit_flag = 1
        for t in self.threads:
            t.join()
        for e in self.q:
            e.close()
        for et in self.r:
            et[1].close()


class Worker(Thread):
    def __init__(self, thread_id, name, q, r, q_lock, r_lock):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.q = q
        self.r = r
        self.q_lock = q_lock
        self.r_lock = r_lock

    def run(self) -> None:
        print("starting", self.name)
        self.reload()
        print("stopping", self.name)

    def reload(self):
        global exit_flag
        while not exit_flag:
            self.q_lock.acquire()
            if not self.q.empty():
                local_env = self.q.get()
                print("got", local_env)
                self.q_lock.release()
                obs = local_env.reset()
                self.r_lock.acquire()
                self.r.put((obs, local_env))
                self.r_lock.release()
            else:
                self.q_lock.release()
                time.sleep(1)
