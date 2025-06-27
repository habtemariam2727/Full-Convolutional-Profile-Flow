#!/usr/bin/env python3

import os
import sys
_parent_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset  
from torch import nn
import wandb
import matplotlib.pyplot as plt

from exp_secondround.cgan.models import Generator, Discriminator
import tools.tools_train as tl

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open('exp_secondround/cgan/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)['UK_cgan']
    
# define the data loader
data_path = os.path.join(_parent_path, 'data/uk_data_cleaned_ind_train.csv')
np_array_train = pd.read_csv(data_path, index_col=0).values

data_path = os.path.join(_parent_path, 'data/uk_data_cleaned_ind_test.csv')
np_array_test = pd.read_csv(data_path, index_col=0).values

# stack one extra column of zeros to the data as the condition
dataloader_train, scaler = tl.create_data_loader(np_array_train, config['batch_size'], True)
# dataloader_test, _ = tl.create_data_loader(np_array_test, config['uk_cgan']['batch_size'], True)

# initialize models
generator = Generator(
    condition_dim=config['condition_dim'],
    noise_dim=config['noise_dim'],
    output_shape=config['output_shape'],
    hidden_dim=config['hidden_dim']
).to(device)


discriminator = Discriminator(
    input_shape=config['output_shape'],
    condition_dim=config['condition_dim'],
    hidden_dim=config['hidden_dim']
).to(device)

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

criterion = nn.BCELoss()

# logging with wandb
wandb.init(project="cgan_uk", config=config)
wandb.watch(generator, log="all")

# log the amount of parameters
print('Generator parameters:', sum(p.numel() for p in generator.parameters() if p.requires_grad))
print('Discriminator parameters:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
wandb.log({
    "generator_params": sum(p.numel() for p in generator.parameters() if p.requires_grad),
    "discriminator_params": sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
})

# training loop
cond_dim = config['condition_dim']
for epoch in range(config['epochs']):
    for _, data in enumerate(dataloader_train):
        generator.train()
        discriminator.train()
        pre = data[0].to(device) # + torch.randn_like(data[0].to(device))/(256)
            
        # split the data into data and conditions
        cond = pre[:,-cond_dim:]
        data = pre[:,:-cond_dim] # + torch.rand_like(data[:,:-cond_dim])/(256) 
        batch_size = data.shape[0]
        
        real_data, condition =  data.to(device), cond.to(device)
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # train discriminator
        optimizer_D.zero_grad()

        outputs_real = discriminator(real_data, condition)
        d_loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(batch_size, config['noise_dim']).to(device)
        fake_data = generator(noise, condition)

        outputs_fake = discriminator(fake_data.detach(), condition)
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # train generator
        optimizer_G.zero_grad()
        outputs_fake = discriminator(fake_data, condition)
        g_loss = criterion(outputs_fake, real_labels)

        g_loss.backward()
        optimizer_G.step()
    
    # plot generated samples
    if (epoch) % 20 == 0:
        generator.eval()
        with torch.no_grad():
            test_data_scaled = scaler.transform(np_array_test)
            condition_test = test_data_scaled[:,-cond_dim:]
            condition = torch.tensor(condition_test).to(device)
            noise = torch.randn(condition.shape[0], config['noise_dim']).to(device)
            fake_data = generator(noise, condition)
            fake_data = torch.concat((fake_data, condition), dim=1)  # concatenate condition to generated data
            fake_data = scaler.inverse_transform(fake_data.cpu().numpy())
            fake_data = fake_data[:, :-cond_dim]  # remove condition for plotting
            plt.figure(figsize=(10, 5))
            plt.plot(fake_data.T, alpha=0.5)
            plt.title(f'Generated Samples at Epoch {epoch + 1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.savefig(f'exp_secondround/cgan/cond_gen_uk/generated_samples.png')
            plt.close()
        
        # save the model
        torch.save(generator.state_dict(), f'exp_secondround/cgan/cond_gen_uk/generator_uk.pth')
        
        # save the geneated samples
        fake_data_df = pd.DataFrame(fake_data)
        fake_data_df.to_csv(f'exp_secondround/cgan/cond_gen_uk/generated_samples.csv', index=False)


    print(f"Epoch [{epoch+1}/{config['epochs']}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
