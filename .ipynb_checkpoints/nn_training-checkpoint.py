import os

import torch
import pandas as pd

if torch.backends.mps.is_available():
    print("nn_training.py is using Metal Performance Shaders API")
    device = torch.device("mps")
else:
    print("Metal Performance Shaders API could not be accessed for nn_training.py. Using CPU instead.")
    device = torch.device("cpu")

def main():

    # Import data
    folder_path = 'single_target_data'
    dataframes = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df_name = f"df_{filename.split('.')[0]}"  
            dataframes[df_name] = pd.read_csv(file_path)

if __name__ == "__main__":
    main()
