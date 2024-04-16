from object import Object

import numpy as np


class Target(Object):

    def __init__(
        self,
        terminal_vel_x: float = 15.0,
        terminal_vel_y: float = 15.0,
        terminal_vel_z: float = 15.0,
    ):
        super().__init__(terminal_vel_x, terminal_vel_y, terminal_vel_z)
