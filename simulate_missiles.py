#! /usr/bin/env python3

"""
Simulate Missiles
    In this script, missile trajectories can be simulated. Goal of this
    Simulation is to try to intercept these missiles using Machine Learning
    algorithms and common methods like the Kalmann filter.

    This script requires Matplotlib.pyplot and the projects Class modules to 
    be installed in the environment its running in.
"""
import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from model import Model
from object import Object
from utils import cart2sphere

EARTH_RADIUS = 6.378 * 10**6


def simulate_missiles(n_missiles, step_ms, d_s, plot, print_pos, print_vel, print_acc):
    """
    Generates the simulation data for the missiles motion

    Parameters
    __________
    n_missiles : int (Not Implemented)
        Sets the number of missiles to be simulated.
    step_ms : int
        The length of intervals between data points in milliseconds
    d_s : int
        Duration of the simulation in seconds
    realtime : bool
        Whether the simulation should run in Realtime
    plot : bool
        Whether the simulation should generate a plot of trajectories after finishing
    """

    # Initialization
    model = Model(np.longdouble(step_ms / 1000))

    # (0,0,0) N
    test_obj = Object(pos_sphere=[EARTH_RADIUS, math.pi / 2, 0], mass=32158, record=True)
    test_obj.set_velocity([3800, 0.0, 4500], system="spherical")
    model.add_Object(test_obj)

    # Preparation
    max_test_height = 0
    max_test_velocity = 0

    # Running
    time_passed_ms = 0
    while time_passed_ms < d_s * 1000:
        # TODO: Implement booster in new missile class

        # Run simulation and record motion
        model.trigger()
        if time_passed_ms % 1000 == 0:
            if print_pos:
                height, lat, long = test_obj.get_coords(system="spherical")
                print(
                    f"Object height: {round(height-EARTH_RADIUS, 2)}     Object coords: ({round(lat, 6)}, {round(long, 6)})"
                )
            if print_vel:
                print(f"Velocity[x, y, z]: {test_obj.vel}")
            if print_acc:
                print(f"Acceleration[x, y, z]: {test_obj.acc}")

        # Record max height
        test_height = cart2sphere(test_obj.pos)[0] - EARTH_RADIUS
        if test_height > max_test_height:
            max_test_height = test_height

        # Record max velocity
        test_velocity = math.sqrt(sum(test_obj.vel[i] ** 2 for i in range(3)))
        if test_velocity > max_test_velocity:
            max_test_velocity = test_velocity

        # Increment the passed time
        time_passed_ms += step_ms

    # Distance calculation, currently doesn't work for movements only along the equator
    start_pos = cart2sphere(
        np.array(
            [
                test_obj.motion_table["pos_x"].iloc[0],
                test_obj.motion_table["pos_y"].iloc[0],
                test_obj.motion_table["pos_z"].iloc[0],
            ]
        )
    )
    end_pos = cart2sphere(
        np.array(
            [
                test_obj.motion_table["pos_x"].iloc[-1],
                test_obj.motion_table["pos_y"].iloc[-1],
                test_obj.motion_table["pos_z"].iloc[-1],
            ]
        )
    )
    dsigma = math.acos(
        math.sin(start_pos[1]) * math.sin(end_pos[1])
        + math.cos(start_pos[1])
        * math.cos(end_pos[1])
        * math.cos(start_pos[2] - end_pos[2])
    )

    print("Flight Statistic")
    print("________________")
    print(f"Distance: {round(dsigma * EARTH_RADIUS / 1000, 1)} km")
    print(f"Apogee: {round(max_test_height / 1000, 1)} km")
    print(f"Max Velocity: {round(max_test_velocity, 1)} m/s")

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        theta = np.linspace(0, np.pi / 2, 100)
        x = EARTH_RADIUS * np.cos(theta)
        y = EARTH_RADIUS * np.sin(theta)
        plt.plot(x, y, label="Quarter Circle")

        ax.plot(
            test_obj.motion_table["pos_x"],
            test_obj.motion_table["pos_y"],
            test_obj.motion_table["pos_z"],
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect("equal")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate object movement")
    parser.add_argument("n_missiles", type=int, help="How many missiles are created")
    parser.add_argument(
        "step_ms",
        type=int,
        help="Timesteps the simulation takes for each calculation in ms",
    )
    parser.add_argument("d_s", type=int, help="How long the simulation runs in seconds")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether a plot of the objects trajectories should be shown",
    )
    parser.add_argument(
        "--pos",
        action="store_true",
        help="Whether the objects position should be printed to console",
    )
    parser.add_argument(
        "--vel",
        action="store_true",
        help="Whether the objects velocity should be printed to console",
    )
    parser.add_argument(
        "--acc",
        action="store_true",
        help="Whether the objects acceleration should be printed to console",
    )

    args = parser.parse_args()

    simulate_missiles(
        args.n_missiles,
        args.step_ms,
        args.d_s,
        args.plot,
        args.pos,
        args.vel,
        args.acc,
    )


if __name__ == "__main__":
    main()
