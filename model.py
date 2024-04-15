from object import Object


gravity = [0, 0, -9.81]


"""

    - obj_container (Array(Object)):
    - obj_state_table (Dictionary(Object, Array)): A dictionary holding the stats of each object in the model. The columns of the state array
        are gravity activated,
"""


class Model:

    def __init__(self, ts_s, level: float = 0):
        self.obj_container = []
        self.obj_state_table = {}
        self.level = level
        self.ts_s = ts_s

    def prepare(self):
        for obj in self.obj_container:
            obj.add_to_acceleration(gravity)
            self.obj_state_table[obj] = [True]

    def trigger(self):
        for obj in self.obj_container:
            obj.trigger(self.ts_s)

            if obj.current_pos[2] < self.level:
                obj.current_pos[2] = self.level
