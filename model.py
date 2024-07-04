from math import sin, cos

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

    def calculate_gravity(self, pos_cart, mass: float) -> np.ndarray:
        """
        Calculates the cartesian force vector for gravity.

        Parameters
        __________
        pos_cart : array-like
            The position of the object in cartesian coordinates.
        mass : float
            The mass of the object in kilograms.

        Returns
        __________
        np.ndarray
            A numpy array representing the cartesian force vector (Fx, Fy, Fz) in Newtons.
        
        """
        pos_cart = np.array(pos_cart)
        r, theta, phi = cart2sphere(pos_cart)
        gravity_force_abs = GRAVITY_CONSTANT * EARTH_MASS * mass / r**2 
        gravity_force_vector_sphere = np.array([-gravity_force_abs, 0.0, 0.0])
        rotation_matrix  = np.array(
            [
                [cos(phi)*sin(theta), cos(phi)*cos(theta), -sin(phi)],
                [sin(phi)*sin(theta), sin(phi)*cos(theta), cos(phi)],
                [cos(theta), -sin(phi), 0],
            ]
        )

        return np.dot(rotation_matrix, gravity_force_vector_sphere)

    def trigger(self):
        """
        Handles the physics simulation. Uses the 4th order Runge-Kutta method to numerically solve the differential equation systems of motion for each object.
        """
        for obj in self.obj_container:
            height, _, _ = obj.get_coords(system="spherical")
            if height < EARTH_RADIUS:
                obj.stop()
                continue
                
            r = obj.get_coords(system="cartesian")
            v = obj.get_velocity(system="cartesian")
            a = obj.get_acceleration(system="cartesian")
            dt = self.step_s
            m = obj.mass

            def acceleration(r):
                gravity = self.calculate_gravity(r, m)
                a_gravity = gravity / m
                return a + a_gravity

            # 4th Order Runge-Kutta method
            k1v = acceleration(r)
            k1r = v

            k2v = acceleration(r + 0.5 * k1r * dt)
            k2r = v + 0.5 * k1v * dt

            k3v = acceleration(r + 0.5 * k2r * dt)
            k3r = v + 0.5 * k2v * dt

            k4v = acceleration(r + k3r * dt)
            k4r = v + k3v * dt

            r_new = r + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
            v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

            obj.set_pos(r_new)
            obj.set_velocity(v_new)

            if obj.record:
                obj.record_motion(dt)

