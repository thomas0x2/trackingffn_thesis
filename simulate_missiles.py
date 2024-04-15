#! /usr/bin/env python3

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from object import Object
from model import Model


def simulate_missiles(n_missiles, step_ms, d_s, realtime, plot):
    # Initialization

    print(realtime)

    model = Model(step_ms / 1000)
    test_obj = Object(pos=[0, 0, 0], vel=[0, 0, 0], acc=[66, 0, 110])
    booster = True
    model.obj_container.append(test_obj)

    # Preparation
    model.prepare()

    # Running
    time_passed_ms = 0
    while time_passed_ms / 1000 < d_s:
        start_time = time.time()

        if booster:
            if time_passed_ms >= 30 * 1000:
                test_obj.add_to_acceleration([-66, 0, -110])
                booster = False

        # Run simulation and record kinematics
        model.trigger(record_kinematics=True)
        if time_passed_ms % 1000 == 0:
            print(test_obj.current_pos)

        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time < step_ms and realtime:
            time.sleep(step_ms / 1000 - execution_time)

        time_passed_ms += step_ms

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            test_obj.kinematics_table["pos_x"],
            test_obj.kinematics_table["pos_y"],
            test_obj.kinematics_table["pos_z"],
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
