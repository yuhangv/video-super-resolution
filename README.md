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
├── notebooks/
│ └── espcn_superres.ipynb # ESPCN implementation + training on Colab
├── scripts/
│ └── train_esrgan_win3070_cli_val.py # ESRGAN trainer with CLI + validation
├── results_val/ # Saved validation results (after training)
├── requirements.txt # Dependencies
└── README.md # This file
