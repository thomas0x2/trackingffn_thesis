import numpy as np

from object import Object


gravity = [0, 0, -9.81]


"""
    The model 
    - obj_container (Array(Object)):
    - obj_state_table (Dictionary(Object, Array)): A dictionary holding the stats of each object in the model. The columns of the state array
        are gravity activated,
"""


class Model:

    def __init__(self, step_s):
        self.obj_container = []
        self.obj_state_table = {}
        self.step_s = step_s

    def prepare(self):
        for obj in self.obj_container:
            obj.add_to_acceleration(gravity)
            self.obj_state_table[obj] = [True]

    def trigger(self, record_kinematics=False):
        for obj in self.obj_container:
            obj.trigger(self.step_s)

            if record_kinematics:
                obj.record_kinematics(self.step_s)

            if obj.current_pos[2] < 0:
                obj.current_pos[2] = 0
                obj.current_vel = np.array([0, 0, 0])
                obj.current_acc = np.array([0, 0, 0])
