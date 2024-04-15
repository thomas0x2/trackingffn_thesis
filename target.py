from object import Object

import numpy as np


class Target(Object):

    def __init__(
        self,
        max_acc_x: float = 15.0,
        max_acc_y: float = 15.0,
        max_acc_z: float = 15.0,
    ):
        super().__init__(max_acc_x, max_acc_y, max_acc_z)
