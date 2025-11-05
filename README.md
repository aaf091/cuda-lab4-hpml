# CUDA Lab 4 — High Performance Machine Learning

This repository contains CUDA implementations for **ECE-GY 9143: High Performance Machine Learning – Lab 4**.  
The assignment explores GPU programming fundamentals, memory models, and performance tuning through three modules:
Vector Operations, Unified Memory, and Convolution.

---

## 📂 Repository Structure

| Folder | Description |
|:--------|:-------------|
| **PartA/** | CUDA matrix & vector operations — coalesced memory and shared-memory matrix multiply |
| **PartB/** | CPU vs GPU performance study using explicit vs unified CUDA memory |
| **PartC/** | 2D convolution (naïve, tiled shared-memory, and cuDNN) with timing and checksum validation |

---

## ⚙️ Build Instructions

All Makefiles support NVIDIA `nvcc`.  
Run each part independently (recommended on GPU clusters such as NYU Greene).

### Example
```bash
cd PartA
make clean && make vecadd01
./vecadd01 1000
```

For Part C (requires cuDNN):
```
cd PartC
module load cuda/12.4 cudnn/8.9
make
./convolution
```
