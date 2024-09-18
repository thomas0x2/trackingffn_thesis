from typing import List

import numpy as np
import pandas as pd

import utils

EARTH_RADIUS = 6.378 * 10**6

class Object:
    """
    A class used to represent any physical object

    Attributes
    __________
    pos : np.ndarray
        current position coordinates
    vel : np.ndarray
        current velocity vector
    acc : np.ndarray
        current acceleration vector
    terminal_vel : np.ndarray
        the terminal velocity when the object stops accelerating
    state_list : list(bool)
        contains the state of the object. Implemented in Object: gravity, collided
    record : bool
        Whether the object should record its motion

    Methods
    _______
    trigger(timestep_s)
        Calculates the states after a timestep
    set_acceleration(acceleration)
        Sets the acceleration to the specified vector
    add_to_acceleration(acceleration)
        Adds the acceleration vector to the current acceleration
    record_motion(timestep_s)
        Adds a record for the next timestep
    """

    def __init__(
        self,
        id: str,
        mass: float,
        c: float,
        A: float,
        pos_sphere = [EARTH_RADIUS, 0, 0],
        pos_cart = None,
        ):
        """
        Parameters
        __________

        pos_sphere : array-like
            The initial position of the object in spherical coordinates (r, theta, phi) or (radius, polar, azimuthal) or (radius, latitude, longitude)
        pos_cart : array-like
            The initial position of the object in cartesian coordinates (x, y, z) where (0, 0, 0) is the center of earth. Defaults to None. If both pos_sphere and pos_cart a given, defaults to pos_cart.
        mass : float
            The mass of the object in kilograms.
        record : bool
            Whether the object should record its motion
        """
        if pos_cart is None:
            pos_sphere = np.array(pos_sphere)
            self.pos = utils.sphere2cart(pos_sphere)
        else:
            pos_cart = np.array(pos_cart)
            self.pos = pos_cart
        self.id = id
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.mass = mass
        self.c = c 
        self.A = A
        self.is_collided = False

    def get_state(self) -> List:
        return [self.pos, self.vel, self.acc]

    def get_properties(self) -> List:
        return [self.mass, self.c, self.A]

    def get_coords(self, system: str = "cartesian") -> np.ndarray:
        """
        Retrieves the coordinates of the object in the specified coordinate system.

        Args:
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".

        Returns:
            np.ndarray: The coordinates of the object in the specified coordinate system.
        """
        if system == "cartesian":
            return self.pos
        elif system == "spherical":
            return utils.cart2sphere(self.pos)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def set_pos(self, pos_vector: np.ndarray, system: str = "cartesian"):
        """
        Sets the position of the object in the specified coordinate system.

        Args:
            pos_vector (np.ndarray): The position vector in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        if system == "cartesian":
            self.pos = pos_vector
        elif system == "spherical":
            self.pos = utils.sphere2cart(pos_vector) 
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def get_velocity(self, system: str = "cartesian") -> np.ndarray:
        if system == "cartesian":
            return self.vel
        elif system == "spherical":
            _, theta, phi = self.get_coords(system="spherical")
            rotation_matrix = utils.rotation_matrix_cartesian(theta, phi)
            return np.linalg.solve(rotation_matrix, self.vel)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def set_velocity(self, vel_vector, system: str = "cartesian"):
        """
        Sets the velocity of the object in the specified coordinate system.

        Args:
            vel_vector (array-like): The velocity vector in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        vel_vector = np.array(vel_vector)
        if system == "cartesian":
            self.vel = vel_vector
        elif system == "spherical":
            _, theta, phi = self.get_coords(system="spherical")
            rotation_matrix = utils.rotation_matrix_cartesian(theta, phi)
            self.vel = np.dot(rotation_matrix, vel_vector)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def get_acceleration(self, system: str = "cartesian") -> np.ndarray:
        if system == "cartesian":
            return self.acc
        elif system == "spherical":
            _, theta, phi = self.get_coords(system="spherical")
            rotation_matrix = utils.rotation_matrix_cartesian(theta, phi)
            return np.linalg.solve(rotation_matrix, self.acc)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def set_acceleration(self, acc_vector, system: str = "cartesian"):
        """
        Sets the acceleration of the object in the specified coordinate system.

        Args:
            acc_vector (np.ndarray): The acceleration vector in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        acc_vector = np.array(acc_vector)
        if system == "cartesian":
            self.acc = acc_vector
        elif system == "spherical":
            _, theta, phi = self.get_coords(system="spherical")
            rotation_matrix = utils.rotation_matrix_cartesian(theta, phi)
            self.acc = np.dot(rotation_matrix, acc_vector)

        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def add_to_acceleration(self, acc_vector, system: str = "cartesian"):
        """
        Adds the given acceleration vector to the current acceleration of the object in the specified coordinate system.

        Args:
            acc_vector (np.ndarray): The acceleration vector to add in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        acc_vector = np.array(acc_vector)
        if system == "cartesian":
            self.acc = self.acc + acc_vector
        elif system == "spherical":
            _, theta, phi = self.get_coords(system="spherical")
            rotation_matrix = utils.rotation_matrix_cartesian(theta, phi)
            self.acc = self.acc + np.dot(rotation_matrix, acc_vector)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def stop(self):
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)

    def record_motion(self, timestep_s):
        """
        Adds an entry to the objects motion table for the next timestep

        Parameters
        __________
        timestep_s : float
            Duration of the timestep for the next entry
        """
        t_s = len(self.motion_table) * timestep_s
        record = pd.DataFrame(
            [
                {
                    "t_s": t_s,
                    "pos_x": self.pos[0],
                    "pos_y": self.pos[1],
                    "pos_z": self.pos[2],
                    "vel_x": self.vel[0],
                    "vel_y": self.vel[1],
                    "vel_z": self.vel[2],
                    "acc_x": self.acc[0],
                    "acc_y": self.acc[1],
                    "acc_z": self.acc[2],
                }
            ]
        )
        self.motion_table = pd.concat([self.motion_table, record])
        self.motion_table.reset_index()
 
    def trigger(self, timestep_s):
        pass

class Booster():
    def __init__(
        self,
        id: str,
        struct_mass: float,
        fuel_mass: float,
        thrust: float,
        mass_flow_rate: float
        ):
            self.id = id
            self.struct_mass = struct_mass
            self.fuel_mass = fuel_mass
            self.total_mass = struct_mass + fuel_mass
            self.thrust = thrust
            self.mass_flow_rate = mass_flow_rate
            self.is_empty = False

    def use_fuel(self, timestep_s):
        if self.is_empty:
            return
        self.fuel_mass = max(self.fuel_mass - self.mass_flow_rate * timestep_s, 0)
        self.total_mass = self.struct_mass + self.fuel_mass
        if self.fuel_mass <= 0:
            self.thrust = 0.0
            self.mass_flow_rate = 0.0
            self.is_empty = True


class Missile(Object):

    def __init__(
        self,
        id: str,
        payload_mass: float,
        c: float,
        A: float,
        boosters: list[Booster],
        angle: list[float],
        pos_sphere = [EARTH_RADIUS, 0, 0],
        pos_cart = None,
        auto_eject: bool = False,
        ):
            mass = payload_mass + sum(booster.total_mass for booster in boosters)
            super().__init__(id, mass, c, A, pos_sphere, pos_cart)
            self.boosters = boosters
            self.payload_mass = payload_mass
            self.time_passed = 0
            self.thrust_direction = self.calculate_thrust_direction(angle)
            self.auto_eject = auto_eject

    def eject_booster(self, booster):
        self.boosters.remove(booster)
        self.mass -= (booster.struct_mass + booster.fuel_mass)

    def calculate_thrust_direction(self, missile_angle):
        _, latitude, longitude = self.get_coords(system="spherical")
        elevation, azimuth = missile_angle

        # Calculate the local up vector (normal to the surface)
        local_up = np.array([
            np.sin(latitude) * np.cos(longitude),
            np.sin(latitude) * np.sin(longitude),
            np.cos(latitude)
        ])

        # Calculate the local east vector
        local_east = np.array([
            -np.sin(longitude),
            np.cos(longitude),
            0
        ])

        # Calculate the local north vector
        local_north = np.cross(local_east, local_up)

        # Calculate the thrust direction in the local coordinate system
        local_thrust = np.array([
            np.sin(elevation) * np.cos(azimuth),
            np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation)
        ])

        # Transform the local thrust to the global coordinate system
        global_thrust = (local_thrust[0] * local_north +
                         local_thrust[1] * local_east +
                         local_thrust[2] * local_up)

        return global_thrust / np.linalg.norm(global_thrust)

    def trigger(self, timestep_s):
        self.time_passed += timestep_s
        total_thrust = 0
        new_booster_masses = 0
        for booster in self.boosters:
            booster.use_fuel(timestep_s)
            new_booster_masses += booster.total_mass
            total_thrust += booster.thrust
        self.mass = self.payload_mass + new_booster_masses
        self.acc = total_thrust / self.mass * self.thrust_direction


    def __str__(self):
        return f"ID: {self.id}\nm: {self.mass} kg\nfuel: {sum(booster.fuel_mass for booster in self.boosters)} kg\ntotal thrust: {sum(booster.thrust for booster in self.boosters)} N"

class Interceptor(Missile):

    def __init__(
        self,
        id: str,
        payload_mass: float,
        c: float,
        A: float,
        boosters: list[Booster],
        angle: list[float],
        pos_sphere = [EARTH_RADIUS, 0, 0],
        pos_cart = None,
        max_lateral_acc = 0,
        ):
        super().__init__(id, payload_mass, c, A, boosters, angle, pos_sphere, pos_cart)
        self.max_lateral_acc = max_lateral_acc

    def set_lateral_acc(self, normal_acc: float, binormal_acc: float):
        pass




