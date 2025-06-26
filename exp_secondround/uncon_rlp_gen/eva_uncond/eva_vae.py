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

vae_gen_datapaht = 'exp_secondround/uncon_rlp_gen/vae/uncond_vae_samples.csv'
diffusion_gen_datapaht = 'exp_secondround/uncon_rlp_gen/diffusion/diff_uncond.csv'
origial_data_path =  os.path.join(_parent_path, 'data', 'ge_data_ind.csv')

# Load the generated data
vae_gen_data = pd.read_csv(vae_gen_datapaht).values
print('vae_gen_data shape: ', vae_gen_data.shape)
# Ensure no negative values
vae_gen_data[vae_gen_data < 0] = 0


# Load the diffusion generated data
diffusion_gen_data = pd.read_csv(diffusion_gen_datapaht).values
print('diffusion_gen_data shape: ', diffusion_gen_data.shape)
diffusion_gen_data[diffusion_gen_data<0] = 0  # Ensure no negative values

# Load the original data
original_data = pd.read_csv(origial_data_path)
original_data = original_data.sample(frac=1).reset_index(drop=True)
original_data = np.array(original_data)
print('original_data shape: ', original_data.shape)

# Plot the samples
save_path_vae = 'exp_secondround/uncon_rlp_gen/eva/vae_samples.png'
save_path_original = 'exp_secondround/uncon_rlp_gen/eva/original_samples.png'
save_path_diffusion = 'exp_secondround/uncon_rlp_gen/eva/diffusion_samples.png'   
pf.plot_consumption(original_data, vae_gen_data, 'VAE', '-', save_path_vae, show_color_bar=True)
pf.plot_consumption(original_data, original_data, 'Original Samples', '-', save_path_original, show_color_bar=True)
pf.plot_consumption(original_data, diffusion_gen_data, 'Diffusion', '-', save_path_diffusion, show_color_bar=True)

# Eva the energy distance
print('Calculating energy distance...')
edis_vae = eva.calculate_energy_distances(original_data, vae_gen_data)
edis_diffusion = eva.calculate_energy_distances(original_data, diffusion_gen_data)
print('Energy distance between original and VAE generated data: ', edis_vae)
print('Energy distance between original and Diffusion generated data: ', edis_diffusion)

# Eva the Wasserstein distance
print('Calculating Wasserstein distance...')
wdis_vae = eva.calculate_w_distances(original_data, vae_gen_data)
wdis_diffusion = eva.calculate_w_distances(original_data, diffusion_gen_data)
print('Wasserstein distance between original and VAE generated data: ', wdis_vae)
print('Wasserstein distance between original and Diffusion generated data: ', wdis_diffusion)

# Eva the autocorrelation MSE
print('Calculating autocorrelation MSE...')
autocorr_mse_vae = eva.calculate_autocorrelation_mse(original_data, vae_gen_data)
autocorr_mse_diffusion = eva.calculate_autocorrelation_mse(original_data, diffusion_gen_data)
print('Autocorrelation MSE between original and VAE generated data: ', autocorr_mse_vae)
print('Autocorrelation MSE between original and Diffusion generated data: ', autocorr_mse_diffusion)

# Eva the KS distance
print('Calculating KS distance...')
ks_distance_vae = eva.ks_distance(original_data, vae_gen_data)
ks_distance_diffusion = eva.ks_distance(original_data, diffusion_gen_data)
print('KS distance between original and VAE generated data: ', ks_distance_vae)
print('KS distance between original and Diffusion generated data: ', ks_distance_diffusion)

# EVA mmd
print('Calculating MMD...')
mmd_vae = eva.MMD_kernel(original_data, vae_gen_data)
mmd_diffusion = eva.MMD_kernel(original_data, diffusion_gen_data)
print('MMD between original and VAE generated data: ', mmd_vae)
print('MMD between original and Diffusion generated data: ', mmd_diffusion)