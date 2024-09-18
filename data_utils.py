import os

import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, data_dir, filename, transform=None, target_transform=None):
        csv_path = os.path.join(data_dir, filename)
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        state = self.data.iloc[idx].values[1:].astype(np.float32)
        label = self.data.iloc[idx+1].values[1:10].astype(np.float32)
        if self.transform is not None:
            state = self.transform(state).values.astype(np.float32)
        if self.target_transform is not None:
            label = self.target_transform(label).values.astype(np.float32)
        return state, label
