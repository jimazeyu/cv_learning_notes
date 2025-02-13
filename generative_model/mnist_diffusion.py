import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

############################################
# 1. Residual-Block U-Net Definition
############################################

class ResidualConvBlock(nn.Module):
    """
    A standard residual convolutional block, similar to ResNet-style blocks.
    - Applies Conv -> BN -> GELU twice,
    - Optionally adds a residual connection (if is_res=True).
    - If in_channels != out_channels, the skip connection is handled by adding
      the output of the second conv to the output of the first conv.
    """
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            # Residual mode
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            # If same channels, add x directly; else add x1
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            
            # Divide by sqrt(2) (~1.414) for stable residual scaling
            return out / 1.414
        else:
            # Non-residual mode
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """
    Downsampling block for a U-Net:
    - ResidualConvBlock + MaxPool2d(2).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)  # downscale by factor of 2
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    """
    Upsampling block for a U-Net:
    - ConvTranspose2d to scale up by factor of 2,
    - Two ResidualConvBlocks,
    - Forward() concatenates skip features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    """
    A generic fully-connected embedding block:
    - Takes an input vector of dimension `input_dim`
    - Outputs a vector of dimension `emb_dim` via a small MLP
    """
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()  # ensure it's float
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """
    A U-Net architecture with:
    - Residual blocks
    - Optional context/label conditioning
    - Time-step embeddings (for diffusion)

    Structure:
    - init_conv -> down1 -> down2 -> to_vec (bottleneck)
    - time/context embeddings
    - up0 -> up1 -> up2 -> final out
    """
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        # Initial conv
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Downsampling
        self.down1 = UnetDown(n_feat, n_feat)        # 28->14
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # 14->7

        # Bottleneck
        self.to_vec = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),  # flatten 7x7 -> 1x1
            nn.GELU()
        )

        # Embeddings for time (t) and context (c)
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        # Upsampling (bottleneck -> 7x7)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=7, stride=7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        
        # 7->14
        self.up1 = UnetUp(4 * n_feat, n_feat)
        # 14->28
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Final output
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, c, t, context_mask):
        """
        x: input image [B, in_channels, 28, 28]
        c: labels [B], used for context embedding
        t: timesteps [B], for time embedding
        context_mask: 0 => use context, 1 => block context
        """
        # -----------------
        # 1) Downsample
        # -----------------
        x_init = self.init_conv(x)
        down1 = self.down1(x_init)      # [B, n_feat, 14, 14]
        down2 = self.down2(down1)       # [B, 2*n_feat, 7, 7]
        hiddenvec = self.to_vec(down2)  # [B, 2*n_feat, 1, 1]

        # -----------------
        # 2) Context Handling
        # -----------------
        # Convert c to one-hot
        c = F.one_hot(c, num_classes=self.n_classes).float()
        # Flip 0<->1 if context_mask=1 => block
        context_mask = context_mask[:, None].repeat(1, self.n_classes)  
        context_mask = -1 * (1 - context_mask)
        c = c * context_mask  # zero out context if blocked

        # -----------------
        # 3) Embeddings
        # -----------------
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # -----------------
        # 4) Upsample path
        # -----------------
        up1 = self.up0(hiddenvec)  # [B, 2*n_feat, 7, 7]
        # Inject embeddings by additive + multiplicative
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # 14x14
        up3 = self.up2(cemb2 * up2 + temb2, down1)  # 28x28

        # -----------------
        # 5) Final output
        # -----------------
        out = self.out(torch.cat([up3, x_init], dim=1))
        return out

############################################
# 2. Diffusion Utilities
############################################

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Creates a linearly increasing beta schedule from beta_start to beta_end.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """
    Extract values from 1-D tensor 'a' at positions 't' and reshape to x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

@torch.no_grad()
def forward_diffusion_sample(x_0, t, alphas_cumprod):
    """
    Diffuse the image x_0 to x_t by adding noise.
    x_0: [B, 1, 28, 28]
    t: [B], timestep
    alphas_cumprod: [T], cumulative product of (1 - beta).
    Returns x_t and the actual noise used.
    """
    sqrt_alphas_cumprod_t = torch.sqrt(extract(alphas_cumprod, t, x_0.shape))
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - extract(alphas_cumprod, t, x_0.shape))
    eps = torch.randn_like(x_0)
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * eps
    return x_t, eps

@torch.no_grad()
def sample(model, image_size, alphas, alphas_cumprod, betas, timesteps, sample_size=16, device="cpu"):
    """
    Reverse diffusion sampling:
    Start from x_T ~ N(0, I), then iteratively predict x_{t-1}.
    We set context_mask=0 => always use context. 
    For unconditional usage, you can pass dummy label c=0 or remove context logic.
    """
    model.eval()
    # Start from noise
    x = torch.randn(sample_size, 1, image_size, image_size).to(device)
    # We'll just pick random labels for demonstration (0..9)
    c = torch.randint(low=0, high=10, size=(sample_size,)).to(device)
    # All context allowed => context_mask=0
    context_mask = torch.zeros_like(c)
    
    for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="Sampling"):
        t = torch.tensor([i]*sample_size, device=device).long()
        # Model predicts noise
        noise_pred = model(x, c, t, context_mask)
        
        alpha_t = extract(alphas, t, x.shape)
        alpha_cumprod_t = extract(alphas_cumprod, t, x.shape)
        beta_t = extract(betas, t, x.shape)

        # x_{t-1} = 1/sqrt(alpha_t) * [x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t)*noise_pred]
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - ( (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) ) * noise_pred
        )
        if i > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = x + sigma_t * z
    
    # Clamp or scale to [-1,1] or [0,1] as needed
    x = x.clamp(-1,1)
    return x

############################################
# 3. Main: Train + Sample
############################################
def main():
    # Hyperparameters
    batch_size = 256
    image_size = 28
    timesteps = 200
    beta_start = 1e-4
    beta_end = 0.02
    num_epochs = 5
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # from [0,1] to [-1,1]
    ])
    dataset = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Beta schedule
    betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # Create model
    model = ContextUnet(in_channels=1, n_feat=128, n_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, labels in pbar:
            x = x.to(device)         # [B,1,28,28]
            labels = labels.to(device)  # [B]
            
            # Pick a random timestep for each sample
            t = torch.randint(low=0, high=timesteps, size=(x.shape[0],), device=device).long()
            
            # context_mask=0 => use label; if you want classifier-free guidance,
            # you can randomly set some mask=1 to block labels
            context_mask = torch.zeros_like(labels)

            # Forward diffusion
            x_noisy, noise = forward_diffusion_sample(x, t, alphas_cumprod)
            
            # Predict the noise with the model
            noise_pred = model(x_noisy, labels, t, context_mask)
            
            # MSE loss between predicted noise and actual noise
            loss = mse(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} done.")

    print("Training complete. Generating samples...")

    # Sampling
    model.eval()
    sampled_images = sample(
        model=model,
        image_size=image_size,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        betas=betas,
        timesteps=timesteps,
        sample_size=16,
        device=device
    )

    # Convert from [-1,1] -> [0,1] for visualization
    sampled_images = (sampled_images + 1) / 2.0

    # Plot first 8 samples
    sampled_images = sampled_images.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(sampled_images[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
