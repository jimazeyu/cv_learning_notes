import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# -------------------------------
# 1. Hyperparameters & Config
# -------------------------------
image_size = 28
timesteps = 200
beta_start = 1e-4
beta_end = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must be identical to the schedule used in training
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

def extract(a, t, x_shape):
    """
    Given a 1D tensor 'a' and a batch of timesteps 't', gather the 
    corresponding values from 'a' and reshape to x_shape for broadcasting.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# -------------------------------
# 2. Model Definition (Must match training)
# -------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, n_feature=64):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n_feature, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feature, n_feature, 3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_feature, n_feature*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feature*2, n_feature*2, 3, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_feature*2, n_feature*2, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(n_feature*2, n_feature, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_feature*2, n_feature, 3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Conv2d(n_feature, 1, 1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_feature*2),
            nn.ReLU(),
            nn.Linear(n_feature*2, n_feature*2)
        )
    
    def forward(self, x, t):
        # Timestep embedding
        t = t.float().unsqueeze(-1)  # [B, 1]
        temb = self.time_embed(t)    # [B, 2*n_feature]
        temb = temb[:, :, None, None]
        
        # Encoder
        x1 = self.conv1(x)                 # [B, n_feature, 28, 28]
        x2 = nn.functional.avg_pool2d(x1, 2)  # [B, n_feature, 14, 14]
        x2 = self.conv2(x2)                # [B, 2*n_feature, 14, 14]
        
        # Bottleneck
        x3 = self.conv3(x2)
        x3 = x3 + temb                      # Incorporate time embedding
        
        # Decoder
        x4 = self.upconv1(x3)              # [B, n_feature, 28, 28]
        x4 = torch.cat([x4, x1], dim=1)     # Skip connection
        x4 = self.conv4(x4)                # [B, n_feature, 28, 28]
        
        out = self.conv5(x4)               # [B, 1, 28, 28]
        return out

# -------------------------------
# 3. Sampling Function (with tqdm)
# -------------------------------
@torch.no_grad()
def sample(model, sample_size=16):
    """
    Reverse diffusion sampling:
    Start from x_T ~ N(0, I) and iteratively predict x_{t-1}.
    Using tqdm to visualize the backward process from T-1 to 0.
    """
    model.eval()
    
    # Start from random noise
    x = torch.randn(sample_size, 1, image_size, image_size).to(device)
    
    # Reverse loop from T-1 down to 0
    for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="Sampling", leave=False):
        t = torch.tensor([i] * sample_size, device=device).long()
        
        # 1. Predict the noise
        noise_pred = model(x, t)
        
        # 2. Compute x_{t-1} (DDPM formula)
        alpha_t = extract(alphas, t, x.shape)
        alpha_cumprod_t = extract(alphas_cumprod, t, x.shape)
        beta_t = extract(betas, t, x.shape)
        
        # x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t)/sqrt(1 - alpha_cumprod_t) * noise_pred )
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
        )
        
        # 3. Add noise (only if i > 0)
        if i > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = x + sigma_t * z
    
    # Clamp to [-1, 1]
    x = x.clamp(-1, 1)
    return x

# -------------------------------
# 4. Main: Load Checkpoint & Generate
# -------------------------------
if __name__ == "__main__":
    # Specify the checkpoint you want to load
    # Could be "latest_checkpoint.pth" or "checkpoints/diffusion_epoch_5.pth"
    checkpoint_path = "latest_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Instantiate the model
    model = SimpleUNet().to(device)
    
    # Load the saved state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    
    # Generate samples
    sample_size = 16
    samples = sample(model, sample_size=sample_size)  # [16, 1, 28, 28]
    
    # Convert from [-1, 1] to [0, 1] for visualization
    samples = (samples + 1) / 2.0
    
    # Show the first 8 images
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
