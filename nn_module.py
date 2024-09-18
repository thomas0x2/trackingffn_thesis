
import torch
import torch.nn as nn 

EARTH_RADIUS = 6.378e+6
DEVICE = torch.device("mps")

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(15, 25),
            nn.Sigmoid(),
            nn.Linear(25, 54),
            nn.Sigmoid(),
            nn.Linear(54, 9),
            nn.Linear(9,9),
        )
        self.x_pred = {}

    def forward(self, x):
        return self.model(x)

    def state_propagation(self, x_0):
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(DEVICE)
        self.x_pred[0] = x_0
        r = x_0[:3]
        r_norm = torch.linalg.vector_norm(r)
        i=1
        while i<=4000:
            old_fuel = self.x_pred[i-1][10]
            new_fuel = max(old_fuel - self.x_pred[i-1][14] * 0.1, 0)
            new_mass = self.x_pred[i-1][9] - old_fuel + new_fuel

            pred = self(self.x_pred[i-1].to(DEVICE))
            self.x_pred[i] = torch.tensor([*pred, new_mass, new_fuel, *self.x_pred[0][11:]], device=DEVICE)

            r = torch.Tensor(self.x_pred[i][:3]).to(DEVICE)
            r_norm = torch.linalg.vector_norm(r)
            i+=1


