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

from vae_model import VAE, loss_function
import tools.tools_train as tl

import matplotlib.pyplot as plt

# define the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp_secondround/uncon_rlp_gen/config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# define the data loader
data_path = os.path.join(_parent_path, 'data', 'ge_data_ind.csv')
np_array = pd.read_csv(data_path).values

# create dataloader
dataloader, scaler = tl.create_data_loader(np_array, config['VAE']['batch_size'], True)

# initialize the VAE model
model = VAE(input_dim=config['VAE']['input_dim'],
            latent_dim=config['VAE']['latent_dim'],
            hidden_dim=config['VAE']['hidden_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config['VAE']['lr'], weight_decay=config['VAE']['weight_decay'])

# training loop
num_epochs = config['VAE']['num_epochs']
model.train()

wandb.init(project="uncond_rlp_gen_vae", config=config)
wandb.watch(model)

# log the amount of parameters
wandb.log({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

init_loss = 33.3
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for _, data in enumerate(dataloader):
        data = data[0].to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
    
    # log the loss to wandb
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    
    #  plot the generated samples every 100 epochs
    if avg_loss < init_loss  and (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            sample = torch.randn(365, config['VAE']['latent_dim']).to(device)
            generated_samples = model.decode(sample).cpu().numpy()
            generated_samples = scaler.inverse_transform(generated_samples)
            
            plt.figure(figsize=(10, 10))
            plt.plot(generated_samples.T, alpha=0.5)
            plt.title(f'Generated Samples at Epoch {epoch + 1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'exp_secondround/uncon_rlp_gen/vae_gen_uncond.png')
            plt.close()

            # save the model 
            path = os.path.join('exp_secondround/uncon_rlp_gen/vae/models', 'uncond_vae_model.pt')
            torch.save(model.state_dict(), path)

            print("Training complete. Model saved.")
            
            # save the data 
            generated_samples_df = pd.DataFrame(generated_samples)
            generated_samples_df.to_csv(os.path.join('exp_secondround/uncon_rlp_gen/vae', 'uncond_vae_samples.csv'), index=False)