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
    
    def _conditional_gmm(self, condition, samples):
        # Generate samples from the GMM model conditioned on the input condition
        samples = self.gmm.sample(n_samples=samples)[0]
        _mean = samples[:, 1].mean(axis=0)
        _percent = condition / _mean
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
        
    def generate(self, 
                 condition_value=5,  # Condition is annual energy consumption in MWh
                 num_samples = 1000): # number of samples to generate
        _conditions = self._conditional_gmm(condition_value, num_samples*365)
        _conditions = _conditions.reshape(num_samples, 365, 2)
        _days_over_year = [i for i in range(365)]
        _days_over_year = np.array(_days_over_year)
        # repeate the days over year for each sample to have shape (num_samples, 365, 1)
        _days_over_year = np.repeat(_days_over_year.reshape(1, 365), num_samples, axis=0)
        _days_over_year = _days_over_year.reshape(num_samples, 365, 1)
        
        # concatenate the days over year with the conditions
        _conditions = np.concatenate((_conditions, _days_over_year), axis=2)
        reshape_condition = _conditions.reshape(num_samples*365, 3)
        reshape_condition = torch.tensor(reshape_condition).float().to(self.device)
        
        # noise
        _noise = torch.randn(num_samples*365, self.config['FCPflow']['num_channels']).to(self.device)
        
        # scaler condition
        reshape_condition = torch.cat((_noise, reshape_condition), dim=1)
        scaled_condition = self.scaler.transform(reshape_condition.cpu().numpy())
        scaled_condition = scaled_condition[:, -3:]
        scaled_condition = torch.tensor(scaled_condition).float().to(self.device)
     
        # Generate samples
        generated_samples = self.generator.inverse(_noise, scaled_condition)
        generated_samples = generated_samples.cpu().detach().numpy()
        generated_samples = np.hstack((generated_samples, scaled_condition.cpu().detach().numpy()))
        generated_samples = self.scaler.inverse_transform(generated_samples)
        
        # _reshape the generated samples
        _values_of_sample = generated_samples[:,:-3]
        _values_of_sample = _values_of_sample.reshape(num_samples, 365*48)
        condition_sampes = generated_samples[:,-3:]
        condition_sampes = condition_sampes.reshape(num_samples, 365, 3)
        return _values_of_sample, condition_sampes

if __name__ == "__main__":
    # Example usage
    # data_path = 'data/uk_data_cleaned_ind_train.csv'
    # data = pd.read_csv(data_path, index_col=0)
    # print(data.head())
    
    generator = DataGenerator(device='cpu')
    samples, _ = generator.generate(condition_value=3, num_samples = 20)
    print(_)
    print(samples.shape)
    
    # plot the generated samples
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 5))
    plt.plot(samples[:], alpha=0.01, c='red')
    plt.title('Generated samples')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('exp/exp_hambt/generated_samples_year.png')
    plt.close()