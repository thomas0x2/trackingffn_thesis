import math

import numpy as np

from object import Object
from utils import cart2sphere, sphere2cart

GRAVITY_CONSTANT = 6.674 * 10**(-11)
EARTH_MASS = 5.974 * 10**24
EARTH_RADIUS = 6.378 * 10**6

class Model:
    """
    Class that simulates the world model and all of its objects

    Attributes
    __________
    obj_container : list(Object)
        Holds all Objects in the World
    obj_state_table : dict(Object, list)
        Holds key value pais of Objects and their state list
    step_s : float
        The timesteps in (fractions of) seconds

    Methods
    _______
    add_Object(object)
        Adds an Object to the `obj_container` and its state list to `obj_state_table`

    prepare()
        Prepares the model and its objects. E.g. applying gravity

    trigger()
        Triggers each object in the `obj_container`, handles Collision
    """

    def __init__(self, step_s):
        """
        Parameters
        __________
        step_s : float
            The timesteps in (fractions of) seconds
        """
        self.obj_container = []
        self.obj_state_table = {}
        self.step_s = step_s

    def add_Object(self, object: Object):
        """
        Adds an Object to the `obj_container` and its state list to `obj_state_table`

        Parameters
        __________
        object : Object
            Object to be added to the model
        """
        self.obj_container.append(object)
        self.obj_state_table[object] = object.state_list

    def prepare(self):
        """
        Prepares the model and its objects. E.g. applying gravity
        """
        pass

    def trigger(self):
        """
        Triggers each object in the `obj_container`, handles Collision
        """
        for obj in self.obj_container:
            r, theta, _ = obj.get_coords(system="spherical")

            gravity_force_abs = GRAVITY_CONSTANT * EARTH_MASS * obj.mass / r**2
            print(gravity_force_abs/obj.mass)
            gravity_force_vector_spherical = np.array([-gravity_force_abs, 0, 0])
            gravity_force_vector_cartesian  = np.dot(obj.conversion_matrix_cartesian(), gravity_force_vector_spherical)

            obj.trigger(self.step_s, force=gravity_force_vector_cartesian)

            if r < EARTH_RADIUS:
                obj.vel = np.array([0, 0, 0])
                obj.acc = np.array([0, 0, 0])
