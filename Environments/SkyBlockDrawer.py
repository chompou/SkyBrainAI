import numpy as np

# str.format('<DrawCuboid x1="%d" y1="4" z1="%d" x2="%d" y2="5" z2="%d" type="leaves"/>'
#           % ((5 + ox), (-3 + oz), (1 + ox), (-7 + oz))) + \

"""
Set of functions to create XML-Draw string used by Minedojo / Malmo to generate a Minecraft world.
"""


def draw_skyblock(ox: int = 0, oz: int = 0):
    """
    Creates a drawstring for one skyblock with provided x-offset and z-offset.
    """
    return \
        str.format('<DrawCuboid x1="%d" y1="4" z1="%d" x2="%d" y2="5" z2="%d" type="leaves"/>'
                   % ((5 + ox), (-3 + oz), (1 + ox), (-7 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="-2" z1="%d" x2="%d" y2="1" z2="%d" type="grass"/>'
                   % ((0 + ox), (0 + oz), (-3 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="-2" z1="%d" x2="%d" y2="1" z2="%d" type="grass"/>'
                   % ((1 + ox), (-3 + oz), (3 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="2" z1="%d" x2="%d" y2="6" z2="%d" type="log"/>'
                   % ((3 + ox), (-5 + oz), (3 + ox), (-5 + oz)))


def get_spawn_positions(ox: int = 0, oz: int = 0):
    """
    Creates a possible spawnpositions for an SkyBblock with provided x-offset and z-offset.
    """

    x_z = np.array([
        (0, 0), (-1, 0), (-2, 0), (-3, 0),
        (0, -1), (-1, -1), (-2, -1), (-3, -1),
        (0, -2), (-1, -2), (-2, -2), (-3, -2),
        (0, -3), (-1, -3), (-2, -3), (-3, -3),
        (0, -4), (-1, -4), (-2, -4), (-3, -4),
        (0, -5), (-1, -5), (-2, -5), (-3, -5),
        (1, -3), (1, -4), (1, -5),
        (1, -3), (2, -3), (3, -3),
    ])

    return x_z + (ox, oz)


def draw_skyblock_grid(rows, cols, marg):
    """
    Creates a XML-Drawstring with a grid of SkyBlock-worlds. This functions can generate a world with many seperate Skyblocks.
    Provide number of rows, columns and the space/margin between each SkyBlock.
    """

    assert marg > 5

    draw_string = ""
    spawn_positions = []

    for i in range(rows):
        for j in range(cols):
            o_x = i * marg
            o_z = j * marg
            draw_string = draw_string + draw_skyblock(o_x, o_z)
            spawn_positions.append(get_spawn_positions(o_x, o_z))

    return draw_string, spawn_positions
