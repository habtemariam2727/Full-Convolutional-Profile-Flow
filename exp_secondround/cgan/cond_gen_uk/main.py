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

from exp_secondround.cgan.models import Generator, Discriminator, compute_gradient_penalty
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
        batch_size = data[0].shape[0]
        generator.train()
        discriminator.train()
        pre = data[0].to(device) # + torch.randn_like(data[0].to(device))/(256)
            
        # split the data into data and conditions
        cond = pre[:,-cond_dim:]
        data = pre[:,:-cond_dim] # + torch.rand_like(data[:,:-cond_dim])/(256) 
        
        real_data, condition =  data.to(device), cond.to(device)
        noise = torch.randn(batch_size, config['noise_dim']).to(device)
        fake_data = generator(noise, condition)
        
        # train discriminator
        optimizer_D.zero_grad()
        
        # compute the gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator,  condition, real_data, fake_data, device)
        
        outputs_real = discriminator(real_data, condition)
        outputs_fake = discriminator(fake_data.detach(), condition)

        d_loss =  -torch.mean(outputs_real) + torch.mean(outputs_fake) + 10 * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # train generator
        for _ in range(1):
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, config['noise_dim']).to(device)
            fake_data = generator(noise, condition)
            
            outputs_fake = discriminator(fake_data, condition)
            
            g_loss = -torch.mean(outputs_fake)  # generator loss    

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
