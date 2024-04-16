import numpy as np
import pandas as pd


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
        pos: np.ndarray = np.zeros(3),
        vel: np.ndarray = np.zeros(3),
        acc: np.ndarray = np.zeros(3),
        terminal_vel: np.ndarray = np.array([150, 150, 150]),
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

        # Check for each argument whether its an np.ndarray, if not try to cast it
        args = {"pos": pos, "vel": vel, "acc": acc, "terminal_vel": terminal_vel}
        for key in args:
            if not isinstance(args[key], np.ndarray):
                try:
                    args[key] = np.array(args[key])
                except TypeError:
                    print(f"{args[key]} is not of type np.ndarray")

        self.pos = args["pos"]
        self.vel = args["vel"]
        self.acc = args["acc"]
        self.terminal_vel = args["terminal_vel"]
        self.state_list = [False, False]

        if record:
            self.record = True
            self.motion_table = pd.DataFrame()

    def trigger(self, timestep_s):
        """
        Calculates the motion for the next timestep. If `self.record` is set to True
        it will also call `self.record_motion()` to append a new record

        Parameters
        __________
        timestep_s : float
            The duration of one timestep in (fractions of) seconds
        """
        t = timestep_s
        self.pos = self.pos + self.vel * t + 0.5 * self.acc * t**2
        self.vel = self.vel + self.acc * t

        if self.record:
            self.record_motion(timestep_s)

    def set_acceleration(self, acceleration: np.ndarray):
        """
        Sets the current acceleration to the passed vector. Expects a np.ndarray or something that can
        be cast into a np.ndarray like a list or a tuple.

        Parameters
        __________
        acceleration : np.ndarray
            The acceleration vector for the object

        Raises
        ______
        TypeError
            If the acceleration parameter is not a np.ndarray and cannot be cast into one
        """
        if type(acceleration) != np.ndarray:
            try:
                acceleration = np.array(acceleration)
            except TypeError:
                print("acceleration is not of type np.ndarray")
        self.acc = acceleration

    def add_to_acceleration(self, acceleration: np.ndarray):
        """
        Adds the passed acceleration vector to the objects current acceleration. Expects a np.ndarray
        or something that can be cast into a np.ndarray like a list or a tuple.

        Parameters
        __________
        acceleration : np.ndarray
            The acceleration vector added to the object

        Raises
        ______
        TypeError
            If the acceleration parameter is not a np.ndarray and cannot be cast into one
        """
        if type(acceleration) != np.ndarray:
            try:
                acceleration = np.array(acceleration)
            except TypeError:
                print("acceleration is not of type np.ndarray")
        new_acceleration = self.acc + acceleration
        self.acc = new_acceleration

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
