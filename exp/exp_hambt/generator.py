import torch
import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, data_path, batch_size, device):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self.data = None
        self.scaler = None

    def load_data(self):
        # Load the data from the CSV file
        self.data = pd.read_csv(self.data_path, index_col=0).values

    def create_data_loader(self):
        # Normalize the data
        self.scaler = (self.data - np.mean(self.data)) / np.std(self.data)

        # Convert to PyTorch tensors and create batches
        data_tensor = torch.tensor(self.scaler, dtype=torch.float32).to(self.device)
        num_batches = len(data_tensor) // self.batch_size

        for i in range(num_batches):
            yield data_tensor[i * self.batch_size:(i + 1) * self.batch_size]