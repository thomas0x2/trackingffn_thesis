#! /usr/bin/env python3

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from object import Object
from model import Model

# Program arguments
STEP_MS = 0.1
REALTIME = False


def simulate_missiles(n_missiles, step_ms, d_s):
    # Initialization
    model = Model(STEP_MS)
    test_obj = Object(pos=[0, 0, 2500], vel=[30, 40, 0], acc=[4, 3, 0])
    model.obj_container.append(test_obj)

    # Preparation
    model.prepare()

    # Running
    time_passed_ms = 0
    while time_passed_ms / 1000 < d_s:
        start_time = time.time()

        # Run simulation and record kinematics
        model.trigger(record_kinematics=True)

        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time < step_ms and REALTIME:
            time.sleep(step_ms / 1000 - execution_time)

        time_passed_ms += step_ms

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

    args = parser.parse_args()

    simulate_missiles(args.n_missiles, args.step_ms, args.d_s)


if __name__ == "__main__":
    main()
