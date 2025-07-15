# fMRI Denoising with 3D U-Net

This project implements a 3D U-Net architecture to denoise fMRI brain volumes. The model is trained on synthetically noised 3D fMRI scans derived from the publicly available [OpenNeuro motor-fMRI dataset](https://openneuro.org/datasets/ds005239). Each 3D volume is treated as an independent sample, and the network learns to recover clean signals from noisy inputs using supervised learning.

🔧 Implemented in PyTorch  
📊 Input shape: `[1, 40, 64, 64]`  
📁 Dataset size used: ~24,000 volumes

"# fMRI_project" 
