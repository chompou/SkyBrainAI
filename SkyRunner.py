import minedojo
import Skyruner_mission_interpreter

image_size = (100, 60)


def create_env():
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

    return Skyruner_mission_interpreter.Mission(survival=True, explore=True, episode_length=100, env=env)
