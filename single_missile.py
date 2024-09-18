#! /usr/bin/env python3

import argparse
import math
import random
import copy
from multiprocessing import Process, Queue
import queue
import traceback
import time

from object import Missile, Booster
from model import Model
from kalman import ExtendedKalmanFilter
import plot as plt
import utils
from nn_module import NeuralNetwork

import torch
import numpy as np
import matplotlib.pyplot as matplt

EARTH_RADIUS = 6.378e+6
DEVICE = torch.device("mps")
MEASUREMENT_FREQ = 10

def init_missile(id: str):
    heavy_payload_mass = 14500
    solid_fuel_booster = Booster(id="SF1", struct_mass=4000, fuel_mass=15e+3, thrust=1.5e+6, mass_flow_rate=400)
    liquid_fuel_booster = Booster(id="LF1", struct_mass=3500, fuel_mass=8e+3, thrust=0.7e+6, mass_flow_rate=200)
    missile_angle = [math.pi/4, math.pi/2]
    booster1= copy.deepcopy(solid_fuel_booster)
    booster2= copy.deepcopy(liquid_fuel_booster)
    missile = Missile(id=id, payload_mass= heavy_payload_mass, boosters=[booster1, booster2], angle=missile_angle, pos_sphere=[EARTH_RADIUS, math.pi/4, 0], c=0.2, A=2)
    return missile

def update_ekf(ekf, measurement_queue, output_queue):
    while True:
        try:
            measurement, measurement_time = measurement_queue.get(block=False)
            if measurement is None and measurement_time is None:
                for state_est in list(ekf.x_pred.values()):
                    output_queue.put(state_est.detach())
                output_queue.put(None)
                break
            ekf.propagation_update(measurement, measurement_time, filter_enabled=True)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"EKF couldn't update due to: {type(e).__name__}: {str(e)}")
            print("Traceback:")
            traceback.print_exc() 

def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    means = checkpoint['means']
    std_devs = checkpoint['std_devs']
    
    model = NeuralNetwork().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, means, std_devs

def standardize(x, means, std_devs):
    return (x-means)/std_devs

def reverse_standardize(x, means, std_devs):
    return x*std_devs + means

def simulate_missile(step_ms: int, plot: bool, print_pos: bool, print_vel: bool, print_acc: bool):

    timestep_s = step_ms / 1000
    model = Model(timestep_s)

    # Object creation
    print("Creating missile instance...")
    missile = init_missile("Heavy SLL")
    model.add_Object(missile)

    # Kalman Filter
    print("Creating filter instance...")
    x_0, y_0, z_0 = utils.sphere2cart([EARTH_RADIUS, math.pi/4, 0])
    x_pred_0 = [x_0, y_0, z_0, 0, 0, 0, 0, 0, 0]
    P_pred_0 = torch.eye(9).double() 
    Q = torch.eye(9)
    R = torch.eye(6)
    ekf = ExtendedKalmanFilter(x_pred_0, P_pred_0, Q, R, timestep_s)
    ekf.init_object(c=0.2, A=2, m=14500+4000+3500+15e+3+8e+3, thrust=1.5e+06+0.7e+6, fuel=15e+3+8e+3, mass_flow_rate=600, orientation_cart=missile.thrust_direction)

    # Neural Network
    print("Creating neural network instance")
    nn, means, std_devs = load_model('models/missile_tracker.pth')

    # Update queue and process
    print("Create filter update process...")
    measurement_queue = Queue()
    output_queue = Queue()
    update_process = Process(target=update_ekf, args=(ekf, measurement_queue, output_queue,))
    print("Instances created")

    # EKF Initial Prediction
    print("Predicting missile trajectory using EKF...")
    t_start = time.time()
    ekf.state_propagation()
    t_end = time.time()
    print(f"Missile trajectory prediction done in {round(t_end - t_start, 4)}")

    # NN Initial Prediction
    print("Predicting missile trajectory using FNN...")
    t_start = time.time()
    x_pred_0 = [*x_pred_0, 14500+4000+3500+15e+3+8e+3, 15e+3+8e+3, 0.2, 2, 1.5e+06+0.7e+6, 600]
    x_pred_0_standardized = standardize(x_pred_0, means, std_devs)
    nn.state_propagation(x_pred_0_standardized)
    t_end = time.time()
    print(f"Missile trajectory prediction done in {round(t_end - t_start, 4)}")

    print("Simulating missile trajectory...")
    time_passed_ms = 0
    trajectory = [missile.get_coords(system="cartesian")]

    update_process.start()

    while not missile.is_collided:
        time_passed_ms += step_ms
        model.trigger()
        pos = missile.get_coords(system="cartesian")
        vel = missile.get_velocity(system="cartesian")
        trajectory.append(pos)
        
        # Gather measurements in queue every 5 seconds
        if time_passed_ms % (MEASUREMENT_FREQ*1000) == 0 and time_passed_ms > 0:
            measurement_queue.put((np.concatenate([pos, vel]), int(time_passed_ms/step_ms)))

        # Print information if required
        if time_passed_ms % 1000 == 0:
            if print_pos:
                height, lat, long = missile.get_coords(system="spherical")
                print(f"Object height: {round(height-EARTH_RADIUS, 2)}     Object coords: ({round(lat, 6)}, {round(long, 6)})")
            if print_vel:
                print(f"Velocity[x, y, z]: {missile.vel}")
            if print_acc:
                print(f"Acceleration[x, y, z]: {missile.acc}")

    distance_col1 = np.linalg.norm(trajectory[-1]) - np.linalg.norm(list(ekf.x_pred.values())[-1])
    print(f"Distance Collision site (actual vs. prediction): {round(distance_col1, 2)} m ")

    measurement_queue.put((None, None))
    print("All measurements taken")
    i = 0
    while True:
        state_est = output_queue.get()
        if state_est is None:
            break
        ekf.x_pred[i] = state_est  
        i += 1

    update_process.join()
    print("Update process has terminated")

    measurement_queue.close()
    measurement_queue.join_thread()
    output_queue.close()
    output_queue.join_thread()


    ekf_estimated_trajectory = [vector[:3].detach().numpy() for vector in list(ekf.x_pred.values())]
    nn_estimated_trajectory = [reverse_standardize(vector.cpu().detach().numpy()[:3], means[:3], std_devs[:3]) for vector in list(nn.x_pred.values())]
    distance_col2 = np.linalg.norm(trajectory[-1]) - np.linalg.norm(ekf_estimated_trajectory[-1])
    print(f"Distance Collision site (actual vs. corrected prediction): {round(distance_col2, 2)} m ")
    


    if plot:
        plt.animate_trajectories([trajectory, ekf_estimated_trajectory, nn_estimated_trajectory], step_ms, time_passed_ms/1000)

    data_real = [np.linalg.norm(trajectory[i]) - EARTH_RADIUS for i in range(len(trajectory))]
    data_kalman = [np.linalg.norm(ekf_estimated_trajectory[i]) - EARTH_RADIUS for i in range(len(ekf_estimated_trajectory))]
    data_nn = [np.linalg.norm(nn_estimated_trajectory[i]) - EARTH_RADIUS for i in range(len(nn_estimated_trajectory))]

    matplt.plot(np.linspace(0, time_passed_ms, len(trajectory)), data_real)
    matplt.plot(np.linspace(0, time_passed_ms, len(ekf_estimated_trajectory)), data_kalman)
    matplt.plot(np.linspace(0, time_passed_ms, len(nn_estimated_trajectory)), data_nn)
    matplt.ylabel('Altitude (m)')
    matplt.xlabel('Time (ms)')
    matplt.title('Altitude vs Time: Real and Kalman Estimated Trajectories')
    matplt.legend(['Real Trajectory', 'Kalman Estimated Trajectory'])
    matplt.grid(True)
    matplt.show()



def main():
    parser = argparse.ArgumentParser(description="Simulate object movement")
    parser.add_argument(
        "step_ms",
        type=int,
        help="Timesteps the simulation takes for each calculation in ms",
    )
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

    simulate_missile(
        args.step_ms,
        args.plot,
        args.pos,
        args.vel,
        args.acc,
    )


if __name__ == "__main__":
    main()
