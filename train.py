from itertools import count
import ddql
import SkyRunner

def train(agent, env, max_episodes=1000000):
    for i_e in range(max_episodes):
        state = env.reset()

        for t in count():
            action = agent.sel_action(state.copy())

            next_state, reward, done, info = env.stepNum(action)

            agent.memory.push(state, action, reward, next_state.copy(), done)

            agent.replay_memory()

            state = next_state

            if done:
                break

def begin():
    mc_env = SkyRunner.create_env()
    steve = ddql.Steve()

    train(steve, mc_env)

begin()