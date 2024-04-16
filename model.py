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
    """
    Class that simulates the world model and all of its objects

    Attributes
    __________
    obj_container : list(Object)
        Holds all Objects in the World
    obj_state_table : dict(Object, list)
        Holds key value pais of Objects and their state list
    step_s : float
        The timesteps in (fractions of) seconds

    Methods
    _______
    add_Object(object)
        Adds an Object to the `obj_container` and its state list to `obj_state_table`

    prepare()
        Prepares the model and its objects. E.g. applying gravity

    trigger()
        Triggers each object in the `obj_container`, handles Collision
    """

    def __init__(self, step_s):
        """
        Parameters
        __________
        step_s : float
            The timesteps in (fractions of) seconds
        """
        self.obj_container = []
        self.obj_state_table = {}
        self.step_s = step_s

    def add_Object(self, object: Object):
        """
        Adds an Object to the `obj_container` and its state list to `obj_state_table`

        Parameters
        __________
        object : Object
            Object to be added to the model
        """
        self.obj_container.append(object)
        self.obj_state_table[object] = object.state_list

    def prepare(self):
        """
        Prepares the model and its objects. E.g. applying gravity
        """
        for obj in self.obj_container:
            obj.add_to_acceleration(gravity)
            self.obj_state_table[obj][0] = True

    def trigger(self):
        """
        Triggers each object in the `obj_container`, handles Collision
        """
        for obj in self.obj_container:
            obj.trigger(self.step_s)

            if obj.pos[2] < 0:
                obj.pos[2] = 0
                obj.vel = np.array([0, 0, 0])
                obj.acc = np.array([0, 0, 0])
