import os
import sys
_parent_path = os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)


import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb
import pandas as pd
import matplotlib.pyplot as plt

from diff_model import FFD, linear_beta_schedule, forward_diffusion_sample
import tools.tools_train as tl

# Sampling from the diffusion model
@torch.no_grad()
def sample(model, sample_shape=(1, 96)):
    x = torch.randn(sample_shape).to(device)
    for t in reversed(range(T)):
        t_batch = torch.full((sample_shape[0],), t, device=device).float() / T
        predicted_noise = model(x, t_batch)
        beta = betas[t]
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        sqrt_recip_alpha = (1 / alpha.sqrt())

        x = sqrt_recip_alpha * (x - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x += beta.sqrt() * noise

    return x.cpu()

# import the configuration
with open(os.path.join(_parent_path,'exp_secondround/uncon_rlp_gen/config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
# Hyperparameters
T = config['Diffusion']['T']  # Number of diffusion steps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
betas = linear_beta_schedule(T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
batch_size = config['Diffusion']['batch_size']

# Model, optimizer, and loss
model = FFD(
        input_dim=config['Diffusion']['input_dim'],
        hidden_dim=config['Diffusion']['hidden_dim'],
        output_dim=config['Diffusion']['output_dim']
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=config['Diffusion']['lr'], weight_decay=config['Diffusion']['weight_decay'])
criterion = nn.MSELoss()

# define the data loader
data_path = os.path.join(_parent_path, 'data', 'ge_data_ind.csv')
np_array = pd.read_csv(data_path).values
dataloader, scaler = tl.create_data_loader(np_array, config['VAE']['batch_size'], True)

# Initialize Weights & Biases
wandb.init(project='diffusion-model', config=config)
wandb.watch(model, log='all')

# log amount of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example training loop (replace with your data loader)
for epoch in range(config['Diffusion']['num_epochs']):  # number of epochs
    model.train()
    for _, data in enumerate(dataloader):
        x_0 = data[0].to(device)
        optimizer.zero_grad()
 
        t = torch.randint(0, T, (x_0.shape[0],)).long().to(device)
        x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        predicted_noise = model(x_noisy, t.float() / T)  # Normalize timestep
        
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        wandb.log({'epoch': epoch, 'loss': loss.item(), 'num_params': num_params})
    
    # plot the sample
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            sampled_data = sample(model, sample_shape=(365, 96))
            scaled_data = scaler.inverse_transform(sampled_data.numpy())
            plt.plot(scaled_data.T, alpha=0.5)
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'exp_secondround/uncon_rlp_gen/diffusion/diff_uncond.png')
            plt.close()
            # save the model 
            torch.save(model.state_dict(), f'exp_secondround/uncon_rlp_gen/diffusion/diff_model.pth')
            
            # save the data
            df = pd.DataFrame(scaled_data)
            df.to_csv(f'exp_secondround/uncon_rlp_gen/diffusion/diff_uncond.csv', index=False)

