from math import exp, sqrt
import time

import torch

G = 6.674 * 10**(-11)
M = 5.974 * 10**24
EARTH_RADIUS = 6.378 * 10**6

def compute_jacobian(func, x):
    x = x.requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda x: func(x), x)

class ExtendedKalmanFilter:

    def __init__(self, x_pred_0, P_pred_0, Q, R, timestep_in_s):
        x_pred_0 = torch.tensor(x_pred_0).detach()

        self.x_pred = {0: x_pred_0}
        self.P_pred = {0: P_pred_0.detach()}
        self.S = {}
        self.W = {}
        self.z = {}
        self.nu = {}

        self.H_k = torch.eye(6, 9).detach().double()
        self.Q_k = Q.detach()
        self.R_k = R.detach()
        self.dt = timestep_in_s

    def init_object(self, c, A, m, thrust, fuel, mass_flow_rate, orientation_cart):
        self.object_c = c
        self.object_A = A
        self.object_m = m
        self.object_thrust = thrust
        self.object_fuel = fuel
        self.object_mfr = mass_flow_rate
        self.object_orientation = orientation_cart

    def state_prediction(self, k):
        return self.state_transition(self.x_pred[k-1], fuel_usage=True).detach()

    def state_prediction_covariance(self, k):
        F_k = compute_jacobian(self.state_transition, self.x_pred[k-1])
        return (torch.matmul(torch.matmul(F_k, self.P_pred[k-1]), F_k.T) + self.Q_k).detach()

    def residual_covariance(self, k):
        return (torch.matmul(torch.matmul(self.H_k, self.P_pred[k-1]), self.H_k.T) + self.R_k).detach()

    def filter_gain(self, k):
        return torch.matmul(torch.matmul(self.P_pred[k], self.H_k.T), torch.linalg.inv(self.S[k])).detach()

    def add_measurement(self, z_k, k):
        self.z[k] = z_k
        self.nu[k] = (z_k - torch.matmul(self.H_k, self.x_pred[k])).detach()

    def update_state_estimate(self, k):
        return (self.x_pred[k] + torch.matmul(self.W[k], self.nu[k])).detach()

    def update_state_covariance(self, k):
        return (self.P_pred[k] - torch.matmul(torch.matmul(self.W[k], self.S[k]), self.W[k].T)).detach()

    def state_transition(self, state, fuel_usage: bool = False):
        """
        Compute the next state using RK4 method.
        state: [r_i, r_j, r_k, v_i, v_j, v_k]
        """
        r = state[:3]
        v = state[3:6]
        a = state[6:]
        c = self.object_c
        A = self.object_A
        m = self.object_m
        d = self.dt
        F = self.object_thrust

        if(fuel_usage):
            if self.object_fuel <= 0:
                a = torch.tensor([0,0,0]).detach()
            else:
                a = torch.tensor(F/m * self.object_orientation).detach()
            used_fuel = self.object_mfr * d
            new_fuel = max(self.object_fuel-used_fuel, 0)
            self.object_m = self.object_m - (self.object_fuel - new_fuel)
            self.object_fuel = new_fuel

        def rho(r):
            h = torch.norm(r) - EARTH_RADIUS
            return 1.225 * exp(-h/8500)
        def acceleration(r, v):
            r_norm = torch.norm(r)
            v_norm = torch.norm(v)
            mach = v_norm / 343
            shock_waves_coef = 1 if mach < 1 else sqrt(1+mach**2)
            return a - 0.5*rho(r)*c*A/m*v_norm*v * shock_waves_coef - G*M*r/r_norm**3

        def rk4_step(r, v, a):
            k1v = acceleration(r, v)
            k1r = v

            k2v = acceleration(r + 0.5*k1r*d, v + 0.5*k1v*d)
            k2r = v + 0.5*k1v*d

            k3v = acceleration(r + 0.5*k2r*d, v + 0.5*k2v*d)
            k3r = v + 0.5*k2v*d

            k4v = acceleration(r + k3r*d, v + k3v*d)
            k4r = v + k3v*d

            v_next = v + d/6*(k1v + 2*k2v + 2*k3v + k4v)
            r_next = r + d/6*(k1r + 2*k2r + 2*k3r + k4r)
            a_next = a

            return torch.cat([r_next, v_next, a_next])

        return rk4_step(r, v, a)

    def state_propagation(self, start_k: int = 1, end_k = None):
        k = start_k
        r_pred = self.x_pred[k-1][:3]
        r_pred_norm = torch.norm(r_pred)
        while r_pred_norm >= EARTH_RADIUS:
            self.x_pred[k] = self.state_prediction(k)
            self.P_pred[k] = self.state_prediction_covariance(k)
            self.S[k] = self.residual_covariance(k)
            self.W[k] = self.filter_gain(k)
            r_pred = self.x_pred[k][:3]
            r_pred_norm = torch.norm(r_pred)
            k += 1
            if end_k is not None:
                if k >= end_k:
                    break
            if r_pred_norm > 2e+8:
                print("Object is predicted to leave earth's orbit! Ending tracking...")
                break

    def propagation_update(self, z_k, k, filter_enabled: bool = True):
        print(f"Measurement update at iteration {k}. Propagating state forward...")
        t_start = time.time()
        z_k = torch.tensor(z_k).detach()
        self.add_measurement(z_k, k)
        if filter_enabled:
            self.x_pred[k] = self.update_state_estimate(k)
            self.P_pred[k] = self.update_state_covariance(k)
        else:
            self.x_pred[k] = torch.cat([z_k, self.x_pred[k][6:]]).detach()
        
        self.state_propagation(start_k=k+1)
        t_end = time.time()
        print(f"Finished state propgation in {round(t_end-t_start, 4)} seconds")
