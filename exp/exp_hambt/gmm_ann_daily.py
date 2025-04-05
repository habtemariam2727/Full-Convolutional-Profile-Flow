from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import pickle


gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
# Load the data
data_path = 'data/uk_data_cleaned_ind_train.csv'
data = pd.read_csv(data_path, index_col=0)
data = data.iloc[:, -2:]
print(data.head(), data.shape)
gmm.fit(data)


# Save the GMM model
model_path = 'exp/exp_hambt/gmm_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(gmm, f)



