#!/usr/bin/env python3

import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import wandb
import yaml
import numpy as np

import alg.models_fcpflow_lin as fcpf
import tools.tools_train as tl

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path, 'exp/exp_hambt/config_uk.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data', 'uk_data_cleaned_ind_train.csv')
# read the data from the second column to the end
np_array_train = pd.read_csv(data_path, index_col=0).values
data_path = os.path.join(_parent_path, 'data', 'uk_data_cleaned_ind_test.csv')
np_array_test = pd.read_csv(data_path,  index_col=0).values

# stack one extra column of zeros to the data as the condition
dataloader_train, scaler = tl.create_data_loader(np_array_train[:,:], config['FCPflow']['batch_size'], True)
dataloader_test, _ = tl.create_data_loader(np_array_test[:,:], config['FCPflow']['batch_size'], True)


# train the model
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['FCPflow']['lr_max'], weight_decay=config['FCPflow']['w_decay'])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=config['FCPflow']['lr_step_size'], 
                                              base_lr=config['FCPflow']['lr_min'], max_lr=config['FCPflow']['lr_max'],
                                              cycle_momentum=False )

# define the wandb
wandb.init(project="fcpflow_uk_hambt")
wandb.watch(model)

# recoded data shape
wandb.config.data_shape = np_array_train.shape
wandb.config.batch_size = config['FCPflow']['batch_size']

# train the model
path = os.path.join(_parent_path, 'exp/exp_hambt')
tl.train(path, model, dataloader_train, optimizer, 4000001, config['FCPflow']['condition_dim'], 
         device, scaler, dataloader_test, scheduler, 100, True)

