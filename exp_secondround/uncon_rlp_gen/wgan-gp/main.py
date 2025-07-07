#!/usr/bin/env python3

import os
import sys
_parent_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import yaml
import numpy as np

import wandb

from wgangp_model import Generator, Discriminator, compute_gradient_penalty
import tools.tools_train as tl

import matplotlib.pyplot as plt

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp_secondround/uncon_rlp_gen/config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)['WGAN_GP']

# define the data loader
data_path = os.path.join(_parent_path, 'data', 'ge_data_ind.csv')
np_array = pd.read_csv(data_path).values

# create dataloader
dataloader, scaler = tl.create_data_loader(np_array, config['batch_size'], True)
    
# initialize models
generator = Generator(
    noise_dim=config['noise_dim'],
    output_shape=config['output_shape'],
    hidden_dim=config['hidden_dim']
).to(device)


discriminator = Discriminator(
    input_shape=config['output_shape'],
    hidden_dim=config['hidden_dim']
).to(device)

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

# logging with wandb
wandb.init(project="cgan_ge", config=config)
wandb.watch(generator, log="all")

# log the amount of parameters
print('Generator parameters:', sum(p.numel() for p in generator.parameters() if p.requires_grad))
print('Discriminator parameters:', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
wandb.log({
    "generator_params": sum(p.numel() for p in generator.parameters() if p.requires_grad),
    "discriminator_params": sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
})

# training loop
for epoch in range(config['epochs']):
    for _, data in enumerate(dataloader):
        batch_size = data[0].shape[0]
        generator.train()
        discriminator.train()
        pre = data[0].to(device) # + torch.randn_like(data[0].to(device))/(256)
            
        # split the data into data and conditions
        data = pre[:,:] # + torch.rand_like(data[:,:-cond_dim])/(256) 
        
        real_data =  data.to(device)
        noise = torch.randn(batch_size, config['noise_dim']).to(device)
        fake_data = generator(noise)
        
        # train discriminator
        optimizer_D.zero_grad()
        
        # compute the gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)
        
        outputs_real = discriminator(real_data)
        outputs_fake = discriminator(fake_data.detach())

        d_loss =  -torch.mean(outputs_real) + torch.mean(outputs_fake) + 10 * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        
        # train generator
        for _ in range(1):
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, config['noise_dim']).to(device)
            fake_data = generator(noise)
            
            outputs_fake = discriminator(fake_data)
            
            g_loss = -torch.mean(outputs_fake)  # generator loss    

            g_loss.backward()
            optimizer_G.step()
            
        # log the losses
        wandb.log({
            "D_loss": d_loss.item(),
            "G_loss": g_loss.item(),
            "epoch": epoch + 1,
        })

            
    
    # # plot generated samples
    # if (epoch) % 20 == 0:
    #     generator.eval()
    #     with torch.no_grad():
    #         # test_data_scaled = scaler.transform( outputs_fake.cpu().numpy())
    #         noise = torch.randn(outputs_fake.shape[0], config['noise_dim']).to(device)
    #         fake_data = generator(noise)
    #         fake_data = scaler.inverse_transform(fake_data.cpu().numpy())
    #         fake_data = fake_data[:, :]  # remove condition for plotting
    #         plt.figure(figsize=(10, 5))
    #         plt.plot(fake_data.T, alpha=0.5)
    #         plt.title(f'Generated Samples at Epoch {epoch + 1}')
    #         plt.xlabel('Time Steps')
    #         plt.ylabel('Values')
    #         plt.savefig(f'exp_secondround/uncon_rlp_gen/wgan-gp/generated_samples.png')
    #         plt.close()
        
    #     # save the model
    #     torch.save(generator.state_dict(), f'exp_secondround/uncon_rlp_gen/wgan-gp/generator_usa.pth')
        
    #     # save the geneated samples
    #     fake_data_df = pd.DataFrame(fake_data)
    #     fake_data_df.to_csv(f'exp_secondround/uncon_rlp_gen/wgan-gp/UNCOND_WGANGP_samples.csv', index=False)


    # print(f"Epoch [{epoch+1}/{config['epochs']}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
