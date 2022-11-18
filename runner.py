import SkyRunner
import ddql
import train
import importlib

env = SkyRunner.create_env()

importlib.reload(ddql)
importlib.reload(train)

agent = ddql.Steve()

train.train(agent, env)

env.quit()
env.reset()
