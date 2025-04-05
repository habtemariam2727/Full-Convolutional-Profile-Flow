import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(_parent_path)

import torch
import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.mixture import GaussianMixture

import alg.models_fcpflow_lin as fcpf
import tools.tools_train as tl

class DataGenerator:
    def __init__(self, device):
        self.device = device
                
        # Load the configuration
        with open('exp/exp_hambt/config_uk.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.config = config    
        
        self._load_gmm_model()
        self._define_model()
        self._load_generator()
        self._load_scaler()

        
    def _define_model(self):
        # train the model
        self.generator = fcpf.FCPflow(self.config['FCPflow']['num_blocks'], self.config['FCPflow']['num_channels'], 
                                self.config['FCPflow']['sfactor'], self.config['FCPflow']['hidden_dim'], self.config['FCPflow']['condition_dim']).to(self.device)
                
    def _load_gmm_model(self, model_path = 'exp/exp_hambt/gmm_model.pkl'):
        # Define the GMM model
        self.gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
        with open(model_path, 'rb') as f:
            self.gmm = pickle.load(f)
    
    def _conditional_gmm(self, condition):
        # Generate samples from the GMM model conditioned on the input condition
        samples = self.gmm.sample(n_samples=1000)[0]
        _mean = samples.mean(axis=0)
        _percent = condition / _mean[0]
        _conditional_samples = samples * _percent
        return _conditional_samples
    
    
    def _load_generator(self, model_path= 'exp/exp_hambt/FCPflow_model.pth'):
        # Load the generator model
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))  
        self.generator.to(self.device)
        self.generator.eval()
            
    def _load_scaler(self, scaler_path = 'exp/exp_hambt/scaler.pkl'):
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
    def generate(self, condition_value=5, num_samples = 1000):
        # Condition is annual energ y consumption in MWh
        _conditions = self._conditional_gmm(condition_value)
        _noise = torch.randn(num_samples, self.config['FCPflow']['num_channels']).to(self.device)
        _conditions = torch.tensor(_conditions).float().to(self.device)
        _conditions = torch.cat((_conditions, _noise), dim=1)
        _scaled_conditions = self.scaler.transform(_conditions.cpu().numpy())
        _scaled_conditions = _scaled_conditions[:,:2]
        _scaled_conditions = torch.tensor(_scaled_conditions).float().to(self.device)
        generated_samples = self.generator.inverse(_noise, _scaled_conditions)
        generated_samples = generated_samples.cpu().detach().numpy()
        generated_samples = np.hstack((_scaled_conditions.cpu().detach().numpy(), generated_samples))
        generated_samples = self.scaler.inverse_transform(generated_samples)
        return generated_samples[:,2:], generated_samples[:,0:2]

if __name__ == "__main__":
    # Example usage
    data_path = 'data/uk_data_cleaned_ind_train.csv'
    data = pd.read_csv(data_path, index_col=0)
    # print(data.head())
    
    generator = DataGenerator(device='cpu')
    samples, _ = generator.generate(condition_value=1, num_samples = 1000)
    print(_)
    print(samples)
    print(samples.mean())