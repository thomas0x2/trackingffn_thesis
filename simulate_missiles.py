#! /usr/bin/env python3

"""
Simulate Missiles
    In this script, missile trajectories can be simulated. Goal of this
    Simulation is to try to intercept these missiles using Machine Learning
    algorithms and common methods like the Kalmann filter.

    This script requires Matplotlib.pyplot and the projects Class modules to 
    be installed in the environment its running in.
"""
import time
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from object import Object
from model import Model


def simulate_missiles(n_missiles, step_ms, d_s, realtime, plot):
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
    model = Model(step_ms / 1000)
    test_obj = Object(pos=[0, 0, 0], vel=[0, 0, 0], acc=[66, 33, 110], record=True)
    booster = True
    model.add_Object(test_obj)

    # Preparation
    model.prepare()
    max_test_height = 0
    max_test_velocity = 0

    # Running
    time_passed_ms = 0
    while time_passed_ms / 1000 < d_s:
        start_time = time.time()

        # TODO: Implement booster in new missile class
        if booster:
            if time_passed_ms >= 30 * 1000:
                test_obj.add_to_acceleration([-66, -33, -110])
                booster = False

        # Run simulation and record motion
        model.trigger()
        if time_passed_ms % 1000 == 0:
            print(test_obj.pos)

        # If running the simulation in real time
        if realtime:
            end_time = time.time()
            execution_time = end_time - start_time
            if execution_time < step_ms:
                time.sleep(step_ms / 1000 - execution_time)

        # Record max height
        test_height = test_obj.pos[2]
        if test_height > max_test_height:
            max_test_height = test_height

        # Record max velocity
        test_velocity = math.sqrt(sum(test_obj.vel[i] ** 2 for i in range(3)))
        if test_velocity > max_test_velocity:
            max_test_velocity = test_velocity

        # Increment the passed time
        time_passed_ms += step_ms

    print("Flight Statistic")
    print("________________")
    print(f"Distance: {round(math.sqrt(test_obj.pos[0]**2 + test_obj.pos[1]**2), 2)} m")
    print(f"Apogee: {round(max_test_height, 2)} m/s")
    print(f"Max Velocity: {round(max_test_velocity, 2)} m/s")

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            test_obj.motion_table["pos_x"],
            test_obj.motion_table["pos_y"],
            test_obj.motion_table["pos_z"],
        )
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
        "--realtime",
        action="store_true",
        help="Whether or not the simulation should run in realtime",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether a plot of the objects trajectories should be shown",
    )

    args = parser.parse_args()

    simulate_missiles(args.n_missiles, args.step_ms, args.d_s, args.realtime, args.plot)


if __name__ == "__main__":
    main()
