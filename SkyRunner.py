import minedojo
import Skyrunner_mission_interpreter

image_size = (600, 650)


def create_env():
    env = minedojo.make(
        "open-ended",
        image_size=image_size,
        generate_world_type='flat',
        flat_world_seed_string="0",
        start_position=dict(x=0, y=2, z=0, yaw=0, pitch=0),
        fast_reset=True,
        start_time=6000,
        allow_time_passage=False,
        drawing_str="""
        <DrawCuboid x1="0" y1="0" z1="0" x2="-3" y2="0" z2="-5" type="dirt"/>
        <DrawCuboid x1="0" y1="-1" z1="0" x2="-3" y2="-1" z2="-5" type="dirt"/>
        <DrawCuboid x1="0" y1="1" z1="0" x2="-3" y2="1" z2="-5" type="grass"/>
        <DrawCuboid x1="1" y1="0" z1="-3" x2="3" y2="0" z2="-5" type="dirt"/>
        <DrawCuboid x1="1" y1="-1" z1="-3" x2="3" y2="-1" z2="-5" type="dirt"/>
        <DrawCuboid x1="1" y1="1" z1="-3" x2="3" y2="1" z2="-5" type="grass"/>
        <DrawCuboid x1="3" y1="2" z1="-5" x2="3" y2="6" z2="-5" type="log"/>
        """
    )

    return Skyrunner_mission_interpreter.Mission(survival=True, explore=True, episode_length=100, min_y=-3,
                                                env=env)
