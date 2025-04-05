import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.mixture import GaussianMixture


class DataGenerator:
    def __init__(self, device):
        self.device = device
        
        # Load gmm model
        self._load_gmm_model()

        
    def _load_gmm_model(self, model_path = 'exp/exp_hambt/gmm_model.pkl'):
        # Define the GMM model
        self.gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
        with open(model_path, 'rb') as f:
            self.gmm = pickle.load(f)

    def _load_generator(self, model_path= 'exp/exp_hambt/FCPflow_model.pth'):
        # Load the generator model
        self.generator = torch.load(model_path, map_location=self.device)
        self.generator.eval()
        
    def _load_scaler(self, scaler_path = 'exp/exp_hambt/scaler.pkl'):
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
    def generate(self, condition=5, num_samples = 1000):
        # Condition is annual energ y consumption in MWh
        pass


if __name__ == "__main__":
    # Example usage
    data_path = 'data/uk_data_cleaned_ind_train.csv'
    data = pd.read_csv(data_path, index_col=0)
    print(data.head())
    
    generator = DataGenerator(device='cpu')