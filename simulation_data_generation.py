import copy
import math
import random

from object import Missile, Booster
from model import Model
import plot as plt

import pandas as pd
import numpy as np

EARTH_RADIUS = 6.378 * 10**6

def init_diverse_missile(id: str, std_dev_angle: float, std_dev_pos: float): 
    # Base values
    base_payload_mass = 10000
    base_solid_booster = Booster(id="SF_base", struct_mass=4000, fuel_mass=15e+3, thrust=1.5e+6, mass_flow_rate=400)
    base_liquid_booster = Booster(id="LF_base", struct_mass=3500, fuel_mass=8e+3, thrust=0.7e+6, mass_flow_rate=200)

    # Randomize parameters
    payload_mass = base_payload_mass * random.uniform(0.9, 1.1) 
    solid_booster = copy.deepcopy(base_solid_booster)
    liquid_booster = copy.deepcopy(base_liquid_booster)

    # Adjust booster parameters
    for booster in [solid_booster, liquid_booster]:
        booster.struct_mass *= random.uniform(0.9, 1.1)
        booster.fuel_mass *= random.uniform(0.9, 1.1)
        booster.thrust *= random.uniform(0.9, 1.1)
        booster.mass_flow_rate *= random.uniform(0.9, 1.1)
 
    # Randomize missile angle
    elevation = math.pi/4 + random.gauss(sigma=std_dev_angle)
    azimuth = math.pi/2 + random.gauss(sigma=std_dev_angle)
    missile_angle = [elevation, azimuth]

    # Randomize starting position on Earth's surface
    latitude = math.pi/4 + random.gauss(sigma=std_dev_pos)
    longitude = random.gauss(sigma=std_dev_pos)
    pos_sphere = [EARTH_RADIUS+5, latitude, longitude]

    # Randomly select 2 or 3 boosters
    num_boosters = 2
    boosters = [copy.deepcopy(solid_booster)]
    boosters.extend([copy.deepcopy(liquid_booster) for _ in range(num_boosters - 1)])

    # Create and return the missile
    missile = Missile(id=id, payload_mass=payload_mass, boosters=boosters, angle=missile_angle, 
                      pos_sphere=pos_sphere, c=random.uniform(0.19, 0.21), A=random.uniform(1.9, 2.1))
    return missile

def generate_missiles(no_missiles):
    missiles = []
    for i in range(no_missiles):
        # Generate a random range factor between 0.5 and 1.5
        
        # Generate the missile name
        missile_name = f"M{i+1}"
        
        # Create the missile entry
        missile = init_diverse_missile(missile_name, std_dev_angle=15e-5, std_dev_pos=1e-5) # 1e-5 equals a position deviation of 628m
        missiles.append(missile)
    
    return missiles

def main():

    missiles = generate_missiles(1000)

    for missile in missiles:
        pos = missile.get_coords(system="cartesian")
        thrust = missile.thrust_direction

        if np.dot(pos, thrust) < 0:
            print(f"Missile {missile.id} is accelerating towards earth!")

    dt_ms = 100
    model = Model(dt_ms/1000)

    columns = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "total_mass", "fuel_mass", "drag_coeff", "frontal_area", "total_thrust", "total_mfr"]
    simulation_data = [pd.DataFrame(columns=pd.Index(columns)) for _ in missiles]
    for missile in missiles:
        model.add_Object(missile)

    static_properties = [[] for _ in range(len(missiles))]
    for i, missile in enumerate(missiles):
        drag_coeff = missile.c
        frontal_area = missile.A
        total_thrust = sum([booster.thrust for booster in missile.boosters])
        total_mfr = sum([booster.mass_flow_rate for booster in missile.boosters])
        static_properties[i] = [drag_coeff, frontal_area, total_thrust, total_mfr]

    time_passed_ms = 0 
    trajectories = [[] for _ in missiles]
    all_missiles_collided = False

    min_simulation_time_ms = 100000  

    while not all_missiles_collided:
        for i, missile in enumerate(missiles):
            if not missile.is_collided:
                simulation_data[i].loc[time_passed_ms] = [time_passed_ms,
                                                          *missile.get_coords(system="cartesian"), 
                                                          *missile.get_velocity(system="cartesian"), 
                                                          *model.get_object_acceleration(missile),
                                                          missile.mass,
                                                          sum(booster.fuel_mass for booster in missile.boosters),
                                                          *static_properties[i]]
                trajectories[i].append(missile.get_coords(system="cartesian"))

        model.trigger()
        time_passed_ms += dt_ms
        all_missiles_collided = all(missile.is_collided for missile in missiles)

        if all_missiles_collided and time_passed_ms >= min_simulation_time_ms:
            break

    valid_indices = [i for i in range(len(missiles)) if len(trajectories[i])>=1000]

    for i in valid_indices:
        dataset = simulation_data[i]
        filename = f"single_target_data/dataset_{i+1}.csv"
        dataset.to_csv(filename, index=False)
        print(f"Saved dataset {i+1} to {filename}")

    plt.animate_trajectories(trajectories, dt_ms, time_passed_ms/1000)




if __name__ == "__main__":
    main()
