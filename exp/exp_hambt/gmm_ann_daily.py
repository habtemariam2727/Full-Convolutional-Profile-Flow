import numpy as np
import pandas as pd
import pickle

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)

# Load the data
data_path = 'data/aus_data_cleaned_annual_habt.csv'
data = pd.read_csv(data_path)
data = data.drop(columns=['days_over_year'])
print(data.head(), data.shape)

gmm.fit(data.iloc[:,-2:])

# Save the GMM model
model_path = 'exp/exp_hambt/gmm_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(gmm, f)
    
# Sample from the GMM model
samples = gmm.sample(n_samples=100)[0]
print(samples.shape)
print(samples[:5])