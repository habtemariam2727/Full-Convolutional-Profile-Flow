import torch
from torch import nn


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.shape[0], 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, noise_dim=12, output_shape=24, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            
            
            nn.Linear(hidden_dim, output_shape)
        )

    def forward(self, noise):
        x = torch.cat([noise], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape=24, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat([x], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Example usage
    noise_dim = 12
    batch_size = 32
    gen = Generator(noise_dim=noise_dim)
    disc = Discriminator()      
    noise = torch.randn(batch_size, noise_dim)
    fake_data = gen(noise)
    disc_out = disc(fake_data)
    print("Generated data shape:", fake_data.shape)
    print("Discriminator output shape:", disc_out.shape)
    
    # Example of computing gradient penalty
    real_data = torch.randn(batch_size, 24)  # Example real data
    grad_penalty = compute_gradient_penalty(disc, real_data, fake_data, device='cpu')
    print("Gradient penalty:", grad_penalty.item())
    print("Generator parameters:", sum(p.numel() for p in gen.parameters() if p.requires_grad))
    print("Discriminator parameters:", sum(p.numel() for p in disc.parameters() if p.requires_grad))