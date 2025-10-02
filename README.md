# Super-Resolution with Deep Learning (ESPCN & ESRGAN)

This repository implements two deep learning approaches for **image and video super-resolution**:
- **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network)  
- **ESRGAN** (Enhanced Super-Resolution GAN)  

Both are trained and evaluated on the **DIV2K dataset**, with support for GPU acceleration, mixed precision, and flexible CLI tools.

---

## 🚀 Highlights
- Implemented **ESPCN** and **ESRGAN** from scratch in PyTorch.  
- Supports **Colab notebooks** for experimentation and reproducibility.  
- GPU optimized: **mixed precision (AMP)**, **batch size scaling**, checkpointing.  
- Automatic **training/validation split** with **DIV2K dataset**.  
- Generates side-by-side comparisons: **LR input | Bicubic | Super-Resolved (SR)**.  

---

## 📂 Repository Structure
~~~
video-super-resolution/
├── notebooks/
│   └── espcn_superres.ipynb        # ESPCN Colab notebook (training + results)
│
├── src/                            # Source code for models & utilities
│   ├── models/
│   │   ├── espcn.py                 # ESPCN architecture
│   │   ├── esrgan.py                # ESRGAN architecture
│   │   └── blocks.py                # Shared layers (RRDB, RDB, etc.)
│   │
│   ├── datasets/
│   │   └── div2k.py                 # DIV2K dataset loader & preprocessing
│   │
│   └── utils/
│       ├── losses.py                # Pixel / perceptual / adversarial losses
│       ├── metrics.py               # PSNR, SSIM, other evaluation metrics
│       └── visualization.py         # Plotting & image saving utilities
│
├── scripts/
│   ├── train_espcn.py               # ESPCN trainer (optional, modularized)
│   └── train_esrgan_win3070_cli_val.py  # ESRGAN trainer with CLI + validation
│
├── results/
│   ├── val/                         # Validation outputs (LR | Bicubic | SR)
│   └── train/                       # (optional) Training visualization samples
│
├── checkpoints/                     # Saved model weights (gitignored)
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
~~~
