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
        radius: float = EARTH_RADIUS,
        theta: float = 0,
        phi: float = 0,
        mass: float = 100,
        record: bool = False,
    ):
        """
        Parameters
        __________

        pos : np.ndarray
            The initial position of the object
        vec : np.ndarray
            The initial velocity vector of the obejct
        acc : np.ndarray
            The initial acceleration of the object
        terminal_vel : np.ndarray
            The terminal velocity when the object stops accelerating
        record : bool
            Whether the object should record its motion
        """
        self.pos = sphere2cart(np.array([radius, theta, phi]))
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.mass = mass

        self.state_list = [False, False]
        if record:
            self.record = True
            self.motion_table = pd.DataFrame()

    def trigger(self, timestep_s: float, force: np.ndarray = np.zeros(3)):
        """
        Calculates the motion for the next timestep. If `self.record` is set to True
        it will also call `self.record_motion()` to append a new record

        Parameters
        __________
        timestep_s : float
            The duration of one timestep in (fractions of) seconds
        """
        t = timestep_s
        add_acc = force / self.mass
        acc = self.acc + add_acc

        self.pos = self.pos + self.vel * t + 0.5 * acc * t**2
        self.vel = self.vel + acc * t

        if self.record:
            self.record_motion(timestep_s)

    def conversion_matrix_cartesian(self) -> np.ndarray:
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

    def set_velocity(self, vel_vector: np.ndarray, system: str = "cartesian"):
        """
        Sets the velocity of the object in the specified coordinate system.

        Args:
            vel_vector (np.ndarray): The velocity vector in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        if system == "cartesian":
            self.vel = vel_vector
        elif system == "spherical":
            self.vel = np.dot(self.conversion_matrix_cartesian(), vel_vector)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def set_acceleration(self, acc_vector: np.ndarray, system: str = "cartesian"):
        """
        Sets the acceleration of the object in the specified coordinate system.

        Args:
            acc_vector (np.ndarray): The acceleration vector in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        if system == "cartesian":
            self.acc = acc_vector
        elif system == "spherical":
            self.acc = np.dot(self.conversion_matrix_cartesian(), acc_vector)

        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

    def add_to_acceleration(self, acc_vector: np.ndarray, system: str = "cartesian"):
        """
        Adds the given acceleration vector to the current acceleration of the object in the specified coordinate system.

        Args:
            acc_vector (np.ndarray): The acceleration vector to add in the specified coordinate system.
            system (str): The coordinate system to use, either "cartesian" or "spherical". Defaults to "cartesian".
        """
        if system == "cartesian":
            self.acc = self.acc + acc_vector
        elif system == "spherical":
            self.acc = self.acc + np.dot(self.conversion_matrix_cartesian(), acc_vector)
        else:
            raise ValueError(
                "Passed system parameter must be 'cartesian' or 'spherical'"
            )

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
        self.motion_table.reset_index(drop=True)
