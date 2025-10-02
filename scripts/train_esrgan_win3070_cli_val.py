
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import requests

# =====================================================
# CLI Arguments
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--amp", type=int, default=1, help="1=use AMP, 0=disable AMP")
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PATCH_SIZE = args.patch_size
USE_AMP = bool(args.amp)

SCALE = 4
HR_DIR = "./DIV2K/DIV2K_train_HR"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | AMP: {USE_AMP} | Batch={BATCH_SIZE}, Patch={PATCH_SIZE}, Epochs={EPOCHS}")

# =====================================================
# Check and Download DIV2K training dataset (HR images only)
# =====================================================
if os.path.isdir(HR_DIR) and len([f for f in os.listdir(HR_DIR) if f.endswith('.png')]) > 0:
    print("Found existing DIV2K training dataset, skipping download.")
else:
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    if not os.path.exists("DIV2K_train_HR.zip"):
        print("Downloading DIV2K training set (~7 GB)...")
        r = requests.get(url, stream=True)
        with open("DIV2K_train_HR.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Extracting DIV2K_train_HR...")
    with zipfile.ZipFile("DIV2K_train_HR.zip", "r") as z:
        z.extractall("./DIV2K")
    print("DIV2K training set ready.")

# =====================================================
# Dataset
# =====================================================
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=4):
        self.files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(".png")]
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.ToTensor()
        ])
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert("RGB")
        hr = self.hr_transform(hr)
        lr = F.interpolate(hr.unsqueeze(0), scale_factor=1/self.scale, mode='bicubic', align_corners=False).squeeze(0)
        return lr, hr

# =====================================================
# ESRGAN Models
# =====================================================
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(channels + i*growth, growth, 3, 1, 1) for i in range(5)
        ])
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.out = nn.Conv2d(channels + 5*growth, channels, 3, 1, 1)

    def forward(self, x):
        inputs = x
        for layer in self.layers:
            out = self.lrelu(layer(inputs))
            inputs = torch.cat([inputs, out], 1)
        return self.out(inputs) * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.rdbs = nn.Sequential(
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
        )
    def forward(self, x): return self.rdbs(x) * 0.2 + x

class Generator(nn.Module):
    def __init__(self, scale_factor=4, num_blocks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.blocks = nn.Sequential(*[RRDB(64) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        up_layers = []
        for _ in range(int(np.log2(scale_factor))):
            up_layers += [nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2)]
        self.upsampler = nn.Sequential(*up_layers)
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)
    def forward(self, x):
        fea = self.conv1(x)
        out = self.blocks(fea)
        out = self.conv2(out) + fea
        out = self.upsampler(out)
        return self.conv3(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in [64,128,256,512]:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        # ✅ Adaptive pooling to make it patch-size agnostic
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1)
        )
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# =====================================================
# Training
# =====================================================
def to_img(t):
    arr = t[0].detach().permute(1,2,0).cpu().numpy()
    return np.clip(arr*255, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    train_dataset = DIV2KDataset(HR_DIR, patch_size=PATCH_SIZE, scale=SCALE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    G = Generator(scale_factor=SCALE).to(device)
    D = Discriminator().to(device)

    pixel_loss = nn.L1Loss()
    vgg = models.vgg19(weights="IMAGENET1K_V1").features[:36].to(device).eval()
    for p in vgg.parameters(): p.requires_grad=False
    def perceptual_loss(sr, hr): return F.l1_loss(vgg(sr), vgg(hr))
    adv_loss = nn.BCEWithLogitsLoss()

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(1, EPOCHS+1):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for lr, hr in loop:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            # Train Discriminator
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                sr = G(lr)
                pred_real = D(hr)
                pred_fake = D(sr.detach())
                d_loss = adv_loss(pred_real, torch.ones_like(pred_real)) + \
                         adv_loss(pred_fake, torch.zeros_like(pred_fake))

            opt_D.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)

            # Train Generator
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred_fake = D(sr)
                g_loss = pixel_loss(sr, hr) + 0.006*perceptual_loss(sr, hr) + \
                         1e-3*adv_loss(pred_fake, torch.ones_like(pred_fake))

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()

            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict()
        }, f"checkpoints/esrgan_epoch{epoch}.pth")
        print(f"Checkpoint saved: checkpoints/esrgan_epoch{epoch}.pth")

    # =====================================================
    # Inference on DIV2K validation set (unseen during training)
    # =====================================================
    val_dir = "./DIV2K/DIV2K_valid_HR"
    if not os.path.isdir(val_dir):
        url_val = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        if not os.path.exists("DIV2K_valid_HR.zip"):
            print("Downloading DIV2K validation set (~300MB)...")
            r = requests.get(url_val, stream=True)
            with open("DIV2K_valid_HR.zip", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extracting DIV2K_valid_HR...")
        with zipfile.ZipFile("DIV2K_valid_HR.zip", "r") as z:
            z.extractall("./DIV2K")
        print("DIV2K validation set ready.")

    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".png")]
    os.makedirs("results_val", exist_ok=True)
    print(f"Running inference on {min(5,len(val_files))} validation images...")

    G.eval()
    for f in val_files[:5]:  # first 5 images
        img = Image.open(f).convert("RGB")
        hr = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Downscale HR to LR
        lr = F.interpolate(hr, scale_factor=1/SCALE, mode="bicubic", align_corners=False)
        bicubic = F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                sr = G(lr)

        # Save comparison
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(to_img(lr)); axs[0].set_title("LR input"); axs[0].axis("off")
        axs[1].imshow(to_img(bicubic)); axs[1].set_title("Bicubic"); axs[1].axis("off")
        axs[2].imshow(to_img(sr)); axs[2].set_title("ESRGAN SR"); axs[2].axis("off")

        out_name = os.path.join("results_val", os.path.basename(f).replace(".png","_sr.png"))
        plt.savefig(out_name)
        plt.close()
        print(f"Saved result: {out_name}")
