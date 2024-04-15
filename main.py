import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from object import Object
from model import Model

# Program arguments
TS_S = 0.1
REALTIME = False

# Initialization
model = Model(TS_S)
test_obj = Object(pos=[0, 0, 2500], vel=[30, 40, 0], acc=[4, 3, 0])
model.obj_container.append(test_obj)

# Preparation
model.prepare()

# Running
iterations = 0
while iterations < 5 / TS_S:
    t_s = round(iterations * TS_S, 2)
    start_time = time.time()
    iterations += 1

    model.trigger()
    test_obj.record_kinematics(t_s)

    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time < TS_S and REALTIME:
        time.sleep(TS_S - execution_time)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plt.plot(
    test_obj.kinematics_table["pos_x"],
    test_obj.kinematics_table["pos_y"],
    test_obj.kinematics_table["pos_z"],
)
plt.show()
