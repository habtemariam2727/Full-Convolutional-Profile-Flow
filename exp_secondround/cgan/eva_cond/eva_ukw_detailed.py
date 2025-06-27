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

gen_datapath = 'exp_secondround/cgan/cond_gen_ukw/generated_samples.csv'
origial_data_path = 'data/uk_data_cleaned_ind_weather_test.csv'

# Load the generated data
gen_data = pd.read_csv(gen_datapath)
print('gen_data shape: ', gen_data.shape)
# Ensure no negative values
gen_data.iloc[:, :-9][gen_data.iloc[:, :-9] < 0] = 0  # Assuming the last two columns are conditions


# Load the original data
original_data = pd.read_csv(origial_data_path, index_col=0).iloc[:, 4:]
print('original_data shape: ', original_data.shape)

gen_data.columns = original_data.columns

print(original_data.head())
print(original_data.columns)
print(gen_data.head())
print(gen_data.columns)

print('--------------MAX TEMP > 25----------------') # Column -5
# filter data of original_data where the maximum temperature is greater than 25
temp_gt_25 = original_data[original_data.iloc[:, -5] > 25]
temp_gt_25_gen = gen_data[gen_data.iloc[:, -5] > 25]
# mmd 
mmd_temp_gt_25 = eva.MMD_kernel(temp_gt_25.iloc[:, :-9].values, temp_gt_25_gen.iloc[:, :-9].values)
# wasserstein distance
wdis_temp_gt_25 = eva.calculate_w_distances(temp_gt_25.iloc[:, :-9].values, temp_gt_25_gen.iloc[:, :-9].values)
# energy distance
edis_temp_gt_25 = eva.calculate_energy_distances(temp_gt_25.iloc[:, :-9].values, temp_gt_25_gen.iloc[:, :-9].values)
# autocorrelation mse
autocorr_mse_temp_gt_25 = eva.calculate_autocorrelation_mse(temp_gt_25.iloc[:, :-9].values, temp_gt_25_gen.iloc[:, :-9].values)
# ks test
ks_temp_gt_25 = eva.ks_distance(temp_gt_25.iloc[:, :-9].values, temp_gt_25_gen.iloc[:, :-9].values)
print(
    'MMD for max temp > 25: ', mmd_temp_gt_25,
    'Wasserstein distance for max temp > 25: ', wdis_temp_gt_25,
    'Energy distance for max temp > 25: ', edis_temp_gt_25,
    'Autocorrelation MSE for max temp > 25: ', autocorr_mse_temp_gt_25,
    'KS test for max temp > 25: ', ks_temp_gt_25
)

print('--------------MIN TEMP < 3----------------') # Column -3
# filter data of original_data where the minimum temperature is less than 3
temp_lt_3 = original_data[original_data.iloc[:, -3] < 3]
temp_lt_3_gen = gen_data[gen_data.iloc[:, -3] < 3]
# mmd 
mmd_temp_lt_3 = eva.MMD_kernel(temp_lt_3.iloc[:, :-9].values, temp_lt_3_gen.iloc[:, :-9].values)
# wasserstein distance
wdis_temp_lt_3 = eva.calculate_w_distances(temp_lt_3.iloc[:, :-9].values, temp_lt_3_gen.iloc[:, :-9].values)
# energy distance
edis_temp_lt_3 = eva.calculate_energy_distances(temp_lt_3.iloc[:, :-9].values, temp_lt_3_gen.iloc[:, :-9].values)
# autocorrelation mse
autocorr_mse_temp_lt_3 = eva.calculate_autocorrelation_mse(temp_lt_3.iloc[:, :-9].values, temp_lt_3_gen.iloc[:, :-9].values)
# ks test
ks_temp_lt_3 = eva.ks_distance(temp_lt_3.iloc[:, :-9].values, temp_lt_3_gen.iloc[:, :-9].values)
print(
    'MMD for min temp < 3: ', mmd_temp_lt_3,
    'Wasserstein distance for min temp < 3: ', wdis_temp_lt_3,
    'Energy distance for min temp < 3: ', edis_temp_lt_3,
    'Autocorrelation MSE for min temp < 3: ', autocorr_mse_temp_lt_3,
    'KS test for min temp < 3: ', ks_temp_lt_3
)

print('--------------IRRADIANCE > 250----------------') # Column -6
# filter data of original_data where the irradiance is greater than 250
irr_gt_250 = original_data[original_data.iloc[:, -6] > 250]
irr_gt_250_gen = gen_data[gen_data.iloc[:, -6] > 250]
# mmd 
mmd_irr_gt_250 = eva.MMD_kernel(irr_gt_250.iloc[:, :-9].values, irr_gt_250_gen.iloc[:, :-9].values)
# wasserstein distance
wdis_irr_gt_250 = eva.calculate_w_distances(irr_gt_250.iloc[:, :-9].values, irr_gt_250_gen.iloc[:, :-9].values)
# energy distance
edis_irr_gt_250 = eva.calculate_energy_distances(irr_gt_250.iloc[:, :-9].values, irr_gt_250_gen.iloc[:, :-9].values)
# autocorrelation mse
autocorr_mse_irr_gt_250 = eva.calculate_autocorrelation_mse(irr_gt_250.iloc[:, :-9].values, irr_gt_250_gen.iloc[:, :-9].values)
# ks test
ks_irr_gt_250 = eva.ks_distance(irr_gt_250.iloc[:, :-9].values, irr_gt_250_gen.iloc[:, :-9].values)
print(
    'MMD for irradiance > 250: ', mmd_irr_gt_250,
    'Wasserstein distance for irradiance > 250: ', wdis_irr_gt_250,
    'Energy distance for irradiance > 250: ', edis_irr_gt_250,
    'Autocorrelation MSE for irradiance > 250: ', autocorr_mse_irr_gt_250,
    'KS test for irradiance > 250: ', ks_irr_gt_250
)   

print('--------------SUNSHINE > 10----------------') # Column -7
# filter data of original_data where the sunshine is greater than 10
sun_gt_10 = original_data[original_data.iloc[:, -7] > 10]
sun_gt_10_gen = gen_data[gen_data.iloc[:, -7] > 10]
# mmd 
mmd_sun_gt_10 = eva.MMD_kernel(sun_gt_10.iloc[:, :-9].values, sun_gt_10_gen.iloc[:, :-9].values)
# wasserstein distance
wdis_sun_gt_10 = eva.calculate_w_distances(sun_gt_10.iloc[:, :-9].values, sun_gt_10_gen.iloc[:, :-9].values)
# energy distance
edis_sun_gt_10 = eva.calculate_energy_distances(sun_gt_10.iloc[:, :-9], sun_gt_10_gen.iloc[:, :-9].values)
# autocorrelation mse
autocorr_mse_sun_gt_10 = eva.calculate_autocorrelation_mse(sun_gt_10.iloc[:, :-9].values, sun_gt_10_gen.iloc[:, :-9].values)
# ks test
ks_sun_gt_10 = eva.ks_distance(sun_gt_10.iloc[:, :-9].values, sun_gt_10_gen.iloc[:, :-9].values)
print(
    'MMD for sunshine > 10: ', mmd_sun_gt_10,
    'Wasserstein distance for sunshine > 10: ', wdis_sun_gt_10,
    'Energy distance for sunshine > 10: ', edis_sun_gt_10,
    'Autocorrelation MSE for sunshine > 10: ', autocorr_mse_sun_gt_10,
    'KS test for sunshine > 10: ', ks_sun_gt_10
)   


print('--------------PRECIPITATION > 10----------------') # Column -2
# filter data of original_data where the precipitation is greater than 10
precip_gt_10 = original_data[original_data.iloc[:, -2] > 10]
precip_gt_10_gen = gen_data[gen_data.iloc[:, -2] > 10]
# mmd   
mmd_precip_gt_10 = eva.MMD_kernel(precip_gt_10.iloc[:, :-9].values, precip_gt_10_gen.iloc[:, :-9].values)
# wasserstein distance
wdis_precip_gt_10 = eva.calculate_w_distances(precip_gt_10.iloc[:, :-9].values, precip_gt_10_gen.iloc[:, :-9].values)
# energy distance
edis_precip_gt_10 = eva.calculate_energy_distances(precip_gt_10.iloc[:, :-9].values, precip_gt_10_gen.iloc[:, :-9].values)
# autocorrelation mse
autocorr_mse_precip_gt_10 = eva.calculate_autocorrelation_mse(precip_gt_10.iloc[:, :-9].values, precip_gt_10_gen.iloc[:, :-9].values)
# ks test
ks_precip_gt_10 = eva.ks_distance(precip_gt_10.iloc[:, :-9].values, precip_gt_10_gen.iloc[:, :-9].values)

print(
    'MMD for precipitation > 10: ', mmd_precip_gt_10,
    'Wasserstein distance for precipitation > 10: ', wdis_precip_gt_10,
    'Energy distance for precipitation > 10: ', edis_precip_gt_10,
    'Autocorrelation MSE for precipitation > 10: ', autocorr_mse_precip_gt_10,
    'KS test for precipitation > 10: ', ks_precip_gt_10
)

