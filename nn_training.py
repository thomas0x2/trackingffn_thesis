import os

from data_utils import CustomDataset
from nn_module import NeuralNetwork

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import pandas as pd 

EARTH_RADIUS = 6.378e+6
DEVICE = torch.device("mps")
EPOCHS = 50
BATCH_SIZE = 64

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for (X, y) in dataloader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f}")
    return test_loss

def load_data(dir_path):
    dataframes = {}
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            df_name = f"df_{filename.split('.')[0]}"  
            dataframes[df_name] = pd.read_csv(file_path)

    return dataframes


def main():
    dataframes = load_data("single_target_data")
    training_data = pd.concat(list(dataframes.values())[:-1])
    means = training_data.iloc[:, 1:].mean()
    std_devs = training_data.iloc[:, 1:].std()

    target_means = means[:9]
    target_std_devs = std_devs[:9]

    def standardize(state):
        state = (state-means)/std_devs
        return state

    def target_standardize(state):
        state = (state-target_means)/target_std_devs
        return state

    Xs = []
    for filename in os.listdir("single_target_data"):
        X = CustomDataset("single_target_data", filename, standardize, target_standardize)
        Xs.append(X)

    X_train = Xs[:-1]
    X_test = Xs[-1]

    print(X_test[0])
    X_train_dl = []
    for X in X_train:
        X_dl = DataLoader(X, batch_size=64, shuffle=True)
        X_train_dl.append(X_dl)

    X_test_dl = DataLoader(X_test, batch_size=64, shuffle=True)

    model = NeuralNetwork().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        for X_dl in X_train_dl:
            train_loop(X_dl, model, loss_fn, optimizer)
        test_loop(X_test_dl, model, loss_fn)

    print("Done!")
    torch.save({"model_state_dict": model.state_dict(),
                "means": means,
                "std_devs": std_devs
                }, 'models/missile_tracker_doublelinear.pth')

if __name__ == "__main__":
    main()
