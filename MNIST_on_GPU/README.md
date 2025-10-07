# MNIST in CUDA  
> From PyTorch to CUDA: step-by-step optimization of an MLP for MNIST

![](assets/mnist-mlp.png)

## Overview
This project implements a **2-layer MLP (784→256→10)** for MNIST digit classification, progressing from **high-level PyTorch** to **low-level CUDA** implementations.  
Each version demonstrates how deep learning computations evolve from Python abstractions to GPU-optimized kernels.

| Version | Framework | Description |
|----------|------------|--------------|
| **v1.py** | PyTorch (CUDA) | Baseline with cuDNN/cuBLAS optimizations |
| **v2.py** | NumPy (CPU) | Manual forward/backward propagation |
| **v3.c** | C (CPU) | Low-level CPU version with timing |
| **v4.cu** | CUDA C | Naive CUDA kernels for matmul, ReLU, softmax |
| **v5.cu** | CUDA + cuBLAS | Optimized version using cuBLAS (SGEMM, SAXPY) |

---

## Setup
```bash
git clone https://github.com/Infatoshi/mnist-cuda
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download MNIST dataset
python downloader.py
