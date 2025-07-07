import torch
from torch import nn


def compute_gradient_penalty(D, cond, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.shape[0], 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, cond)
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
    def __init__(self, condition_dim=12, noise_dim=12, output_shape=24, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(condition_dim + noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dim, output_shape)
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape=24, condition_dim=12, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape + condition_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)


if __name__ == "__main__":
    noise_dim = 12
    condition_dim = 12
    batch_size = 16

    gen = Generator(condition_dim=condition_dim, noise_dim=noise_dim)
    disc = Discriminator(condition_dim=condition_dim)

    noise = torch.randn(batch_size, noise_dim)
    condition = torch.randn(batch_size, condition_dim)

    fake_data = gen(noise, condition)
    disc_out = disc(fake_data, condition)

    print("Generated data shape:", fake_data.shape)
    print("Discriminator output shape:", disc_out.shape)
    
    # Example of computing gradient penalty
    real_data = torch.randn(batch_size, 24)  # Example real data
    grad_penalty = compute_gradient_penalty(disc, condition, real_data, fake_data, device='cpu')
    print("Gradient penalty:", grad_penalty.item())
    print("Generator parameters:", sum(p.numel() for p in gen.parameters() if p.requires_grad))
