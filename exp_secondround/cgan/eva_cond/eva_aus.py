import os
import sys
_parent_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as plt

import plot_function as pf
import evaluation_m as eva

gen_datapath = 'exp_secondround/cgan/cond_gen_aus/generated_samples.csv'
origial_data_path = os.path.join(_parent_path, 'data/aus_data_cleaned_annual_test.csv')

# Load the generated data
gen_data = pd.read_csv(gen_datapath).values
print('gen_data shape: ', gen_data.shape)
# Ensure no negative values
gen_data[gen_data < 0] = 0

# drop nan and inf values
gen_data = gen_data[~np.isnan(gen_data).any(axis=1)]
gen_data = gen_data[~np.isinf(gen_data).any(axis=1)]
print('gen_data shape after dropping nan and inf: ', gen_data.shape)

# Load the original data
original_data = pd.read_csv(origial_data_path, index_col=0)
original_data = original_data.sample(frac=1).reset_index(drop=True)
original_data = np.array(original_data.iloc[:, 2:])
print('original_data shape: ', original_data.shape)

# drop nan and inf values
original_data = original_data[~np.isnan(original_data).any(axis=1)]
original_data = original_data[~np.isinf(original_data).any(axis=1)]
print('original_data shape after dropping nan and inf: ', original_data.shape)

# Plot the samples
save_path = 'exp_secondround/cgan/eva_cond/aus_gen_samples.png'
save_path_original = 'exp_secondround/cgan/eva_cond/aus_original_samples.png'  
pf.plot_consumption(original_data[:,:-2], gen_data, 'cGAN', '-', save_path, show_color_bar=True)
pf.plot_consumption(original_data[:,:-2], original_data[:,:-2], 'Original Samples', '-', save_path_original, show_color_bar=True)

# Eva the energy distance
print('Calculating energy distance...')
edis_vae = eva.calculate_energy_distances(original_data[:,:-2], gen_data)
print('Energy distance between original and cGAN generated data: ', edis_vae)

# Eva the Wasserstein distance
print('Calculating Wasserstein distance...')
wdis_vae = eva.calculate_w_distances(original_data[:,:-2], gen_data)
print('Wasserstein distance between original and cGAN generated data: ', wdis_vae)

# Eva the autocorrelation MSE
print('Calculating autocorrelation MSE...')
autocorr_mse_vae = eva.calculate_autocorrelation_mse(original_data[:,:-2], gen_data)
print('Autocorrelation MSE between original and cGAN generated data: ', autocorr_mse_vae)

# Eva the KS distance
print('Calculating KS distance...')
ks_distance_vae = eva.ks_distance(original_data[:,:-2], gen_data)
print('KS distance between original and VAE generated data: ', ks_distance_vae)

# EVA mmd
print('Calculating MMD...')
mmd_vae = eva.MMD_kernel(original_data[:,:-2], gen_data)
print('MMD between original and cGAN generated data: ', mmd_vae)
