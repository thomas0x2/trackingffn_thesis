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
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button

from model import Model
from object import Object, Missile, Booster
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

    # Model Initialization
    model = Model(np.longdouble(step_ms / 1000))

    # Object creation
    light_payload_mass = 8200
    heavy_payload_mass = 14500
    solid_fuel_booster = Booster(id="SF1", struct_mass=4500, fuel_mass=18*10**3, thrust=1.7*10**6, mass_flow_rate=450)
    liquid_fuel_booster = Booster(id="LF1", struct_mass=3800, fuel_mass=9*10**3, thrust=0.8*10**6, mass_flow_rate=220)
    missile_angle = [math.pi/4, math.pi/2]
    std_dev = 3

    objects = []
    for i in range(n_missiles):
        rnd = random.random()
        rnd_angle = [random.gauss(sigma=std_dev), random.gauss(sigma=std_dev)]
        missile_angle += rnd_angle
        if (rnd>0.5):
            booster1 = copy.deepcopy(solid_fuel_booster)
            booster2 = copy.deepcopy(solid_fuel_booster)
            object = Missile(id="LPSF", payload_mass = light_payload_mass, boosters=[booster1, booster2], angle=missile_angle, pos_sphere=[EARTH_RADIUS, math.pi/2, 0], record=True)
        else:
            booster1= copy.deepcopy(solid_fuel_booster)
            booster2= copy.deepcopy(liquid_fuel_booster)
            booster3= copy.deepcopy(liquid_fuel_booster)
            object = Missile(id="HPLF", payload_mass= heavy_payload_mass, boosters=[booster1, booster2, booster3], angle=missile_angle, pos_sphere=[EARTH_RADIUS, math.pi/2, 0], record=True)
        objects.append(object)
        model.add_Object(object)


    time_passed_ms = 0
    all_trajectories = [[] for _ in range(n_missiles)]

    while time_passed_ms < d_s * 1000:
        model.trigger()
        
        for i, obj in enumerate(objects):
            pos = obj.get_coords(system="cartesian")
            all_trajectories[i].append(pos)

        # Print information if required
        if time_passed_ms % 1000 == 0:
            if print_pos:
                height, lat, long = objects[0].get_coords(system="spherical")
                print(f"Object height: {round(height-EARTH_RADIUS, 2)}     Object coords: ({round(lat, 6)}, {round(long, 6)})")
            if print_vel:
                print(f"Velocity[x, y, z]: {objects[0].vel}")
            if print_acc:
                print(f"Acceleration[x, y, z]: {objects[0].acc}")

        time_passed_ms += step_ms

    if plot:
        animate_trajectories(all_trajectories, step_ms, d_s)

    """
        # Record max height
        test_height = cart2sphere(test_obj.pos)[0] - EARTH_RADIUS
        if test_height > max_test_height:
            max_test_height = test_height

        # Record max velocity
        test_velocity = math.sqrt(sum(test_obj.vel[i] ** 2 for i in range(3)))
        if test_velocity > max_test_velocity:
            max_test_velocity = test_velocity

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
    """




def animate_trajectories(all_trajectories, step_ms, d_s):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create sphere (Earth)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = EARTH_RADIUS * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS * np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Prepare missile lines
    missile_lines = [ax.plot([], [], [], 'r-')[0] for _ in range(len(all_trajectories))]

    # Set initial view
    ax.set_xlim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_ylim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_zlim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Missile Trajectories')

    # Animation control
    paused = False
    current_frame = [0]

    def update(frame):
        if not paused:
            current_frame[0] = frame
            for i, trajectory in enumerate(all_trajectories):
                if frame < len(trajectory):
                    x_data, y_data, z_data = zip(*trajectory[:frame+1])
                    missile_lines[i].set_data(x_data, y_data)
                    missile_lines[i].set_3d_properties(z_data)
        return missile_lines

    def on_click(event):
        nonlocal paused
        paused = not paused

    def on_press(event):
        if event.key == 'right' and current_frame[0] < len(all_trajectories[0]) - 1:
            current_frame[0] += 1
        elif event.key == 'left' and current_frame[0] > 0:
            current_frame[0] -= 1
        update(current_frame[0])
        fig.canvas.draw()

    # Add pause button
    pause_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    pause_button = Button(pause_ax, 'Pause/Play')
    pause_button.on_clicked(on_click)

    fig.canvas.mpl_connect('key_press_event', on_press)

    anim = animation.FuncAnimation(fig, update, frames=len(all_trajectories[0]), 
                                   interval=step_ms/5000, blit=False, repeat=False)

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
