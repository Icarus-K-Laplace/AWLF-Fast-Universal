# AWLF-Fast-Universal
High-performance, JIT-compiled denoising engine for uncooled thermal imaging. Features 60+ FPS processing via Numba/SIMD, native 16-bit RAW support, and robust dead pixel correction. GPL-3.0 licensed; designed for industrial inspection, thermography, and embedded edge deployment.
# AWLF-Fast: Real-time Industrial Infrared Image Restoration

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Powered%20by-Numba-orange.svg)](https://numba.pydata.org/)

**AWLF-Fast** is a high-performance, JIT-compiled implementation of the Adaptive Weighted Local Fitting algorithm, designed specifically for **Uncooled Thermal Imaging (Long-wave IR)**.

Unlike the [Reference Implementation](https://github.com/Icarus-K-Laplace/Adaptive-Weighted-Local-Fitting-AWLF-Denoiser), this engine is optimized for **Production Environments**.
## 🧩 Algorithm Pipeline

![AWLF-Fast Pipeline](./Untitled diagram-2026-01-20-105901.png)

**Key Innovations:**
*   **Numba JIT Compilation**: Critical loops are compiled to native machine code, achieving 60+ FPS on 1024x1024 images.
*   **Adaptive Weighting**: Blending between polynomial fitting (texture) and median (noise) based on local gradient.
*   **16-bit Native Support**: Direct processing of industrial thermal sensor outputs without precision loss.

## 🚀 Key Features

*   **⚡ Blazing Fast**: Achieves **60+ FPS** on standard CPU (i7-10th gen) for 640x512 video streams.
*   **🏭 Industrial Ready**: Native support for **16-bit RAW / TIFF** data (no dynamic range loss).
*   **🧵 Parallelized**: Fully utilizes multi-core CPUs via SIMD vectorization and OpenMP.
*   **🛡️ Robust**: Handles dead pixels (blind pixels) and stripe noise common in IR sensors.
## 👁️ Visual Results & Performance

We evaluate AWLF-Fast on standard infrared benchmarks under varying impulse noise densities (20%, 40%, 60%).

### 1. Global Comparison (Robustness Test)
> **Observation**: The algorithm demonstrates exceptional robustness across all noise levels. Even at **60% noise density**, where traditional median filters fail (causing "salt-and-pepper" artifacts), AWLF-Fast successfully reconstructs the global structure.

![Global Comparison](multi_noise_summary.png)

---

### 2. Zoom-in Details (Texture Preservation)
> **Key Feature**: Unlike standard filters that blur edges, AWLF-Fast preserves sharp transitions (e.g., hat brim, facial features) using adaptive local fitting.

#### Case A: Low Density (20%)
*Ideal for routine sensor calibration.*
![Zoom 20%](roi_detail_20pct.png)

#### Case B: Medium Density (40%)
*Simulating degraded sensor conditions.*
![Zoom 40%](roi_detail_40pct.png)

#### Case C: Extreme Density (60%)
*Stress test for industrial failure scenarios. Note the clean reconstruction of the eye region.*
![Zoom 60%](roi_detail_60pct.png)

## 📊 Performance Benchmark

| Resolution | Implementation | FPS | Latency |
| :--- | :--- | :--- | :--- |
| **640x512 (VGA)** | Pure Python (Ref) | 0.8 fps | 1250 ms |
| **640x512 (VGA)** | **AWLF-Fast** | **72 fps** | **13 ms** |
| **1280x1024 (HD)**| **AWLF-Fast** | **25 fps** | **40 ms** |

> *Tested on Intel Core i7-10750H, Single Thread vs Multi-Thread.*
## 📊 Performance Benchmark

Tested on `scene1.png` (512×640 infrared image):

| Noise Density | Mode        | Time (s) | PSNR Gain | SSIM  |
|:-------------:|:-----------:|:--------:|:---------:|:-----:|
| **20%**       | Auto Detect | **0.4s** | +21.54 dB | 0.980 |
| **20%**       | Full Mask   | **0.4s** | +27.38 dB | 0.985 |
| **40%**       | Auto Detect | **0.8s** | +21.32 dB | 0.947 |
| **40%**       | Full Mask   | **0.8s** | +25.34 dB | 0.960 |
| **60%**       | Auto Detect | **1.4s** | +20.26 dB | 0.877 |
| **60%**       | Full Mask   | **1.1s** | +23.38 dB | 0.911 |

---
## Dataset

Experiments are conducted on the **Tianjin University open UAV remote-sensing infrared grayscale image dataset**.

- Dataset ID (CSTR): `CSTR:14804.11.sciencedb.space.00579`
- Link: https://cstr.cn/14804.11.sciencedb.space.00579
- Note: This repository does **not** redistribute the dataset files. Please download it from the official source and comply with its license/terms.
## 🛠️ Installation

```bash
pip install -r requirements.txt
# Or install as a package
pip install .
