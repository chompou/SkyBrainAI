from itertools import count
import numpy as np
import ddql
import minedojo

SCREEN_RESOLUTION = (64, 64)

def train(agent, env, max_episodes=1000000):
    for i_e in range(max_episodes):
        state = env.reset()

        for t in count():
            action = agent.sel_action(state['rgb'].copy())

            next_state, reward, done, info = env.step(agent.translate_action(action))

            agent.memory.push(state['rgb'].copy(), action, reward, next_state['rgb'].copy(), done)

            agent.replay_memory()

            state = next_state

            env.render()

            if done:
                break

def begin():
    mc_env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=SCREEN_RESOLUTION,
        fast_reset=True,
    )
    steve = ddql.Steve()

    train(steve, mc_env)

begin()