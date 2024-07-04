from math import cos, sin

import numpy as np
import pandas as pd

from utils import cart2sphere, sphere2cart

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
        pos_sphere = [EARTH_RADIUS, 0, 0],
        pos_cart = None,
        mass: float = 100,
        record: bool = False,
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
            self.pos = sphere2cart(pos_sphere)
        else:
            pos_cart = np.array(pos_cart)
            self.pos = pos_cart
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.mass = mass

        if record:
            self.record = True
            self.motion_table = pd.DataFrame()

    def conversion_matrix_cartesian(self) -> np.ndarray:
        """
        Returns the conversion matrix to convert spherical velocity or acceleration vectors to cartesian.
        """
        _, theta, phi = cart2sphere(self.pos)
        matrix = np.array(
            [
                [cos(phi)*sin(theta), cos(phi)*cos(theta), -sin(phi)],
                [sin(phi)*sin(theta), sin(phi)*cos(theta), cos(phi)],
                [cos(theta), -sin(phi), 0],
            ]
        )
        return matrix

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
            return cart2sphere(self.pos)
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
            self.pos = sphere2cart(pos_vector) 
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def get_velocity(self, system: str = "cartesian") -> np.ndarray:
        if system == "cartesian":
            return self.vel
        elif system == "spherical":
            rotation_matrix = self.conversion_matrix_cartesian()
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
            self.vel = np.dot(self.conversion_matrix_cartesian(), vel_vector)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def get_acceleration(self, system: str = "cartesian") -> np.ndarray:
        if system == "cartesian":
            return self.acc
        elif system == "spherical":
            rotation_matrix = self.conversion_matrix_cartesian()
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
            self.acc = np.dot(self.conversion_matrix_cartesian(), acc_vector)

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
            self.acc = self.acc + np.dot(self.conversion_matrix_cartesian(), acc_vector)
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
