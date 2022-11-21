import numpy as np


def draw_skyblock(ox: int = 0, oz: int = 0):
    return \
        str.format('<DrawCuboid x1="%d" y1="0" z1="%d" x2="%d" y2="0" z2="%d" type="dirt"/>'
                   % ((0 + ox), (0 + oz), (-2 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="-1" z1="%d" x2="%d" y2="-1" z2="%d" type="dirt"/>'
                   % ((0 + ox), (0 + oz), (-2 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="1" z1="%d" x2="%d" y2="1" z2="%d" type="grass"/>'
                   % ((0 + ox), (0 + oz), (-2 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="0" z1="%d" x2="%d" y2="0" z2="%d" type="dirt"/>'
                   % ((1 + ox), (-3 + oz), (3 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="-1" z1="%d" x2="%d" y2="-1" z2="%d" type="dirt"/>'
                   % ((1 + ox), (-3 + oz), (3 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="1" z1="%d" x2="%d" y2="1" z2="%d" type="grass"/>'
                   % ((1 + ox), (-3 + oz), (3 + ox), (-5 + oz))) + \
        str.format('<DrawCuboid x1="%d" y1="2" z1="%d" x2="%d" y2="5" z2="%d" type="log"/>'
                   % ((3 + ox), (-5 + oz), (3 + ox), (-5 + oz)))


def get_spawn_positions(ox: int = 0, oz: int = 0):
    x_z = np.array([
        (0, 0), (-1, 0), (-2, 0),
        (0, -1), (-1, -1), (-2, -1),
        (0, -2), (-1, -2), (-2, -2),
        (0, -3), (-1, -3), (-2, -3),
        (0, -4), (-1, -4), (-2, -4),
        (0, -5), (-1, -5), (-2, -5),
        (1, -3), (1, -4), (1, -5),
        (2, -3), (3, -3),
    ])

    return x_z + (ox, oz)


def draw_skyblock_grid(rows, cols, marg):
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
