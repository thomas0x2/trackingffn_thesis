import numpy as np
import pandas as pd


class Object:

    def __init__(
        self,
        pos: np.ndarray = np.zeros(3),
        vel: np.ndarray = np.zeros(3),
        acc: np.ndarray = np.zeros(3),
        max_acc: np.ndarray = np.array([15.0, 15.0, 15.0]),
    ):
        args = {"pos": pos, "vel": vel, "acc": acc, "max_acc": max_acc}

        for key in args:
            if not isinstance(args[key], np.ndarray):
                try:
                    args[key] = np.array(args[key])
                except TypeError:
                    print(f"{args[key]} is not of type np.ndarray")

        self.current_pos = args["pos"]
        self.current_vel = args["vel"]
        self.current_acc = args["acc"]
        self.max_acc = args["max_acc"]

        self.kinematics_table = pd.DataFrame()

    def trigger(self, timestep_s):
        t = timestep_s

        self.current_pos = (
            self.current_pos + self.current_vel * t + 0.5 * self.current_acc * t**2
        )
        self.current_vel = self.current_vel + self.current_acc * t

    def set_acceleration(self, acceleration: np.ndarray):
        if type(acceleration) != np.ndarray:
            try:
                acceleration = np.array(acceleration)
            except TypeError:
                print("acceleration is not of type np.ndarray")
        self.current_acc = acceleration

    def add_to_acceleration(self, acceleration: np.ndarray):
        if type(acceleration) != np.ndarray:
            try:
                acceleration = np.array(acceleration)
            except TypeError:
                print("acceleration is not of type np.ndarray")
        new_acceleration = self.current_acc + acceleration
        self.current_acc = new_acceleration

    def record_kinematics(self, step_s):
        t_s = len(self.kinematics_table) * step_s
        record = pd.DataFrame(
            [
                {
                    "t_s": t_s,
                    "pos_x": self.current_pos[0],
                    "pos_y": self.current_pos[1],
                    "pos_z": self.current_pos[2],
                    "vel_x": self.current_vel[0],
                    "vel_y": self.current_vel[1],
                    "vel_z": self.current_vel[2],
                    "acc_x": self.current_acc[0],
                    "acc_y": self.current_acc[1],
                    "acc_z": self.current_acc[2],
                }
            ]
        )
        self.kinematics_table = pd.concat([self.kinematics_table, record])
        self.kinematics_table.reset_index(drop=True)
