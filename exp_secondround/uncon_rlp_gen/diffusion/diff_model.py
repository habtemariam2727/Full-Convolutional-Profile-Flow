import torch
import torch.nn as nn
import torch.optim as optim

# Simple feed-forward network for time-series (1x96)
class FFD(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, output_dim=96):
        super(FFD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, t):
        # Concatenate timestep embedding
        t_embed = t.unsqueeze(1) #.repeat(1, x.shape[0])
        x_t = torch.cat([x, t_embed], dim=1)
        return self.net(x_t)

# Diffusion schedule utilities
def linear_beta_schedule(T):
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start, beta_end, T)

# Forward diffusion process: add noise
def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_0)
    x_t = sqrt_alphas_cumprod[t][:, None] * x_0 + sqrt_one_minus_alphas_cumprod[t][:, None] * noise
    return x_t, noise


if __name__ == "__main__":
    # Example usage
    T = 100  # Number of diffusion steps
    betas = linear_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Create model
    model = FFD(input_dim=96, hidden_dim=128, output_dim=96)

    # Example input
    x_0 = torch.randn(32, 96)  # Batch of 32 samples
    t = torch.randint(0, T, (32,))  # Random timesteps

    # Forward diffusion sample
    x_t, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    
    print("x_t shape:", x_t.shape)
    print("Noise shape:", noise.shape)