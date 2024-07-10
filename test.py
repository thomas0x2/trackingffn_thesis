#! /usr/bin/env python3
import object, model
import unittest
from utils import sphere2cart

EARTH_RADIUS = 6.378 * 10**6

def run_simulation(duration_s: int):
    t_ms = 50
    sim_env = model.Model(t_ms/1000)
    test_obj = object.Object(record=True)
    test_obj.set_acceleration([0, 0, 15])
    sim_env.add_Object(test_obj)
    
    t_passed = 0 
    while t_passed < duration_s*1000:
        sim_env.trigger()
        t_passed += t_ms

    return [test_obj.pos, test_obj.vel, test_obj.acc]


class MotionTest(unittest.TestCase):

    def test_pos(self):
        test_pos_0, _, _= run_simulation(0)
        test_pos_1, _, _ = run_simulation(1)
        test_pos_10, _, _ = run_simulation(10)

# Very small delta to account for tiny inaccuracies
        delta = 1e-3

# Test for 0 seconds
        self.assertAlmostEqual(test_pos_0[0], 0, delta=delta)
        self.assertAlmostEqual(test_pos_0[1], 0, delta=delta)
        self.assertAlmostEqual(test_pos_0[2], EARTH_RADIUS, delta=delta)

# Test for 1 second
        self.assertAlmostEqual(test_pos_1[0], 0, delta=delta)
        self.assertAlmostEqual(test_pos_1[1], 0, delta=delta)
        self.assertAlmostEqual(test_pos_1[2], EARTH_RADIUS + 2.600, delta=delta)

# Test for 10 seconds
        self.assertAlmostEqual(test_pos_10[0], 0, delta=delta*10)
        self.assertAlmostEqual(test_pos_10[1], 0, delta=delta*10)
        self.assertAlmostEqual(test_pos_10[2], EARTH_RADIUS + 259.950, delta=delta*10)

    def test_vel(self):
        _, test_vel_0, _ = run_simulation(0)
        _, test_vel_1, _ = run_simulation(1)
        
        delta = 1e-3

        self.assertAlmostEqual(test_vel_0[0], 0, delta=delta)
        self.assertAlmostEqual(test_vel_0[1], 0, delta=delta)
        self.assertAlmostEqual(test_vel_0[2], 0, delta=delta)

        self.assertAlmostEqual(test_vel_1[0], 0, delta=delta)
        self.assertAlmostEqual(test_vel_1[1], 0, delta=delta)
        self.assertAlmostEqual(test_vel_1[2], 5.199, delta=delta)


    def test_acc(self):
        _, _, test_acc_0 = run_simulation(0)
        _, _, test_acc_1 = run_simulation(1)
        
        self.assertEqual(test_acc_0[0], 0)
        self.assertEqual(test_acc_0[1], 0)
        self.assertEqual(test_acc_0[2], 15)
        
        self.assertEqual(test_acc_1[0], 0)
        self.assertEqual(test_acc_1[1], 0)
        self.assertEqual(test_acc_1[2], 15)

    def test_gravity(self):
        sim_env = model.Model(0.1)
        pos = sphere2cart([EARTH_RADIUS, 0, 0])
        gravity = sim_env.calculate_gravity(pos, 1)

        self.assertEqual(gravity[0], 0)
        self.assertEqual(gravity[1], 0)
        self.assertAlmostEqual(gravity[2], -9.801, delta=0.001)



if __name__ == "__main__":
    unittest.main()

