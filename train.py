import time
from datetime import datetime
from itertools import count
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import ddql
import SkyRunner


def train(agent, env, max_episodes=1000000, checkpoint_every=10, update_stats_every=1, render=False):
    identifier = str(datetime.now())

    for i_e in range(max_episodes):
        state = env.reset()
        accumulated_reward = 0
        elapsed_sim_time = []

        for e_c in count():
            start_time_sim = time.perf_counter()
            action = agent.sel_action(state)
            next_state, reward, done, info = env.stepNum(action)
            elapsed_sim_time.append(time.perf_counter() - start_time_sim)

            agent.memory.push(state, action, reward, next_state, done)
            accumulated_reward += reward

            agent.replay_memory()
            elapsed_sim_time.append(time.perf_counter() - start_time_sim)

            state = next_state

            if render:
                env.render()

            if done:

                if (i_e + 1) % checkpoint_every == 0:
                    agent.save_checkpoint(identifier + '-' + str(last_num_avg(10)))

                if (i_e + 1) % update_stats_every == 0:
                    add_to_plot(i_e,
                                accumulated_reward,
                                np.array(elapsed_sim_time).mean(),
                                "./" + identifier + "/" + str(i_e) + ".chckp"
                                )

                break


plot_history = []


def last_num_avg(num=10):
    val = 0
    for i in plot_history[len(plot_history) - 10:]:
        val += i[1]
    return val / num


def add_to_plot(e, r, sim_time, identifier="generic_id"):
    plot_history.append((e, r, sim_time))
    render_plot_history()
    # plt.savefig(identifier + "/train_history.png")


def render_plot_history():
    clear_output()
    plt.clf()

    hist = np.array(plot_history)

    e = hist[:, 0]
    reward = hist[:, 1]
    sim_time = hist[:, 2]

    # Reward
    plt.subplot(211)
    plt.plot(e, reward, label='reward')
    plt.title = 'Accumulated Reward'
    plt.legend()

    # Simulation time
    plt.subplot(212)
    plt.plot(e, sim_time, label='simulation_time')
    plt.title = 'Average Simulation time'
    plt.legend()

    plt.show()


def begin():
    mc_env = SkyRunner.create_env()
    steve = ddql.Steve()

    train(steve, mc_env)
