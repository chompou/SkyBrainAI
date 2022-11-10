import minedojo
import Skyruner_mission_interpreter

image_size = (100, 60)

env = minedojo.make(
    "open-ended",
    image_size=image_size,
    generate_world_type='flat',
    flat_world_seed_string="0",
    start_position=dict(x=0, y=2, z=0, yaw=0, pitch=0),
    fast_reset=True,
    drawing_str="""
    <DrawCuboid x1="0" y1="0" z1="0" x2="-2" y2="0" z2="-2" type="dirt"/>
    <DrawCuboid x1="0" y1="-1" z1="0" x2="-2" y2="-1" z2="-2" type="dirt"/>
    <DrawCuboid x1="0" y1="1" z1="0" x2="-2" y2="1" z2="-2" type="grass"/>
    """
)

mission = Skyruner_mission_interpreter.Mission(survival=True, explore=True, episode_length=100, env=env)
mission.reset()

for j in range(5):
    for i in range(3):
        _, reward, __, ___ = mission.stepNum(7)
        print(reward)
    for k in range(10):
        _, reward, __, ___ = mission.stepNum(4)
    for y in range(10):
        done = False
        reward = 0
        while not done:
            _, rev1, done, ___ = mission.stepNum(0)
            reward += rev1
        print("dead", reward)
        mission.reset()
    env.reset()

