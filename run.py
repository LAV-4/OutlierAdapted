from utils import *
from modules import UNet_conditional, EMA
import torch
import torch.nn.functional as F
from ddpm_conditional import Diffusion

if __name__ == '__main__':
    print("Start run")
    n = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = UNet_conditional(num_classes=10).to(device)
    print("Loading model checkpoint...")
    ckpt = torch.load("conditional_ema_ckpt.pt")
    print("Checkpoint loaded successfully.")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=3)
    plot_images(x)