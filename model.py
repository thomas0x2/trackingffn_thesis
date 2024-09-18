from math import exp

import numpy as np

from object import Object
from utils import cart2sphere

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
        self.time_passed = 0

    def add_Object(self, object: Object):
        """
        Adds an Object to the `obj_container` and its state list to `obj_state_table`

        Parameters
        __________
        object : Object
            Object to be added to the model
        """
        self.obj_container.append(object)

    def calculate_gravity(self, pos_cart, mass):
        pos_cart = np.array(pos_cart)
        r = np.linalg.norm(pos_cart)
        gravity_force_abs = GRAVITY_CONSTANT * EARTH_MASS * mass / r**2
        return gravity_force_abs * (-pos_cart / r)

    def calculate_drag(self, pos, vel, c, A):
        pos, vel = map(np.array, (pos, vel))
        v_abs = np.linalg.norm(vel)
        if v_abs == 0:
            return np.zeros(3)
        
        height = cart2sphere(pos)[0] - EARTH_RADIUS
        air_density = 1.225 * exp(-height / 8500)
        drag_abs = 0.5 * air_density * v_abs**2 * c * A
        
        mach = v_abs / 343
        if mach > 1:
            drag_abs *= np.sqrt(1 + mach**2)
        
        return drag_abs * (-vel / v_abs)

    def acceleration(self, r, v, a, m, c, A):
        return a + self.calculate_gravity(r, m) / m + self.calculate_drag(r, v, c, A) / m

    def calculate_runge_kutta_step(self, obj):
        dt = self.step_s
        r, v, a = obj.get_state()
        m, c, A = obj.get_properties()

        def k(r, v):
            return v, self.acceleration(r, v, a, m, c, A)

        k1r, k1v = k(r, v)
        k2r, k2v = k(r + 0.5 * k1r * dt, v + 0.5 * k1v * dt)
        k3r, k3v = k(r + 0.5 * k2r * dt, v + 0.5 * k2v * dt)
        k4r, k4v = k(r + k3r * dt, v + k3v * dt)

        r_new = r + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

        return r_new, v_new    

    def get_object_acceleration(self, obj) -> np.ndarray:
        r, v, a  = obj.get_state()
        m, c, A = obj.get_properties()
        return self.acceleration(r, v, a, m, c, A)

    def trigger(self):
        """
        Handles the physics simulation. Uses the 4th order Runge-Kutta method to numerically solve the differential equation systems of motion for each object.
        """
        self.time_passed += self.step_s
        for obj in self.obj_container:
            height, _, _ = obj.get_coords(system="spherical")
            if height < EARTH_RADIUS:
                if not obj.is_collided:
                    obj.stop()
                    print(f"{obj.id} collided at t={self.time_passed}")
                    obj.is_collided = True
                continue
            obj.trigger(self.step_s)
            r_new, v_new = self.calculate_runge_kutta_step(obj)
            obj.set_pos(r_new)
            obj.set_velocity(v_new)

