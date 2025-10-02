import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fcpflow_package import FCPflowPipeline

# Initialize the pipeline
pipeline = FCPflowPipeline(
    num_blocks = 2,  # Number of blocks in the model
    num_channels = 24,  # Resolution of the time series 
    hidden_dim = 12,  # Dimension of the hidden layers
    condition_dim = 2,  # Dimension of the condition vector, must be larger than 1
    sfactor = 0.3,  # Scaling factor
)

# Define the save path
save_path = ''  # Directory where model and outputs will be saved

# Prepare the data (this data is provided in path 'data/nl_data_1household.csv' in the repository)
data_path = r'nl_data_1household.csv'
data_array = pd.read_csv(data_path).iloc[:, 3:-2].values # data array
condition_array = pd.read_csv(data_path).iloc[:, -2:].values # condition array
np_array = np.hstack((data_array, condition_array)) # concate
np_array = np_array[~pd.isna(np_array).any(axis=1)] # cancel nan

# Define the learning set and the model 
pipeline._define_learning_set()
pipeline._define_model()

# Train the model
num_iter = 801 # number of iterations
save_path = ''  # path to save the model and the figures
val_array = None # validation data
train_scheduler = False # whether to use the learning rate scheduler
pipeline.train_model(num_iter, np_array[:20, :], val_array, save_path, device='cpu', train_scheduler=train_scheduler)

# Load the trained model (If you have a trained model, you can directly load it)
model_path = save_path + 'FCPflow_model.pth'
model = pipeline.load_model(model_path)

# This step is necessary as we scale the data in training
# In this step we fit a scaler
pipeline.data_processing(np_array, None)

# Sample from the trained model based on the conditions
condition_array = np_array[:10, -2:]
samples = pipeline.sample_from_trained_model(condition_array, device='cpu')

# Plot the samples
plt.plot(samples[:, :-2].T)
plt.savefig(save_path + 'sample.png')