from math import sin, cos, exp, sqrt

import numpy as np

from object import Object
from utils import cart2sphere, sphere2cart, rotation_matrix_cartesian

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
        gravity_direction = np.array([-1, 0, 0])
        gravity_force = gravity_force_abs * gravity_direction
        rotation_matrix  = rotation_matrix_cartesian(theta, phi)

        return np.dot(rotation_matrix, gravity_force)

    def calculate_drag(self, pos, vel, drag_coefficient: float = 0.25, area: float = 2):
        """
        Calculates the drag force that applies to the object in Newtons as a cartesian vector (Fx, Fy, Fz).
        """
        pos = np.array(pos)
        vel = np.array(vel)
        v_abs = np.linalg.norm(vel)
        if v_abs == 0:
            return 0 
        height, _, _ = cart2sphere(pos) - EARTH_RADIUS
        
        rho_0 = 1.225
        scale_height = 8500
        air_density = rho_0 * exp(-height/scale_height)

        drag_abs = 0.5 * air_density * v_abs**2 * drag_coefficient * area
        drag_direction = -vel / v_abs

        return drag_abs * drag_direction

    def acceleration(self, r, v, a, m) -> np.ndarray:
        gravity = self.calculate_gravity(r, m)
        a_gravity = gravity / m

        drag = self.calculate_drag(r, v)
        a_drag = drag / m

        return a + a_gravity + a_drag



    def calculate_runge_kutta_step(self, obj: Object, dt: float):
        r, v, a, m = obj.get_state()

        k1v = self.acceleration(r, v, a, m)
        k1r = v

        k2v = self.acceleration(r + 0.5 * k1r * dt, v + 0.5 * k1v * dt, a, m)
        k2r = v + 0.5 * k1v * dt

        k3v = self.acceleration(r + 0.5 * k2r * dt, v + 0.5 * k2v * dt, a, m)
        k3r = v + 0.5 * k2v * dt

        k4v = self.acceleration(r + k3r * dt, v + k3v * dt, a, m)
        k4r = v + k3v * dt

        r_new = r + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

        return r_new, v_new



    def trigger(self):
        """
        Handles the physics simulation. Uses the 4th order Runge-Kutta method to numerically solve the differential equation systems of motion for each object.
        """
        for obj in self.obj_container:
            height, _, _ = obj.get_coords(system="spherical")
            if height < EARTH_RADIUS:
                obj.stop()
                continue

            dt = self.step_s
            r_new, v_new = self.calculate_runge_kutta_step(obj, dt)
            obj.set_pos(r_new)
            obj.set_velocity(v_new)

            if obj.record:
                obj.record_motion(dt)

