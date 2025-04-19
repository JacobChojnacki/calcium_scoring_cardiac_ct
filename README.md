# Automatic Plaque Segmentation & Agatston Scoring from Non‑Contrast Cardiac CT

This repository provides the code and resources to reproduce the experiments from:
> *The comparison of 2D and 3D based models for the problem of plaque segmentation and coronary artery calcium scoring on non‑contrast cardiac CT imaging* :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

## 🔍 Overview

Coronary Artery Calcium (CAC) scoring is a key biomarker for cardiovascular risk assessment. Manual quantification of CAC via the Agatston score is time‑consuming and subject to inter‑observer variability. This project implements and evaluates several 2D and 3D deep‐learning segmentation architectures for fully‑automatic plaque detection and subsequent Agatston score estimation on low‑dose, non‑contrast CT scans :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}.

---

## 🚀 Features

- **2D vs 3D architectures**  
  - **UNet2D**, **RaUNet2D**, **UNETR2D**  
  - **UNet3D**, **RaUNet3D**  
- **Preprocessing pipeline**  
  - HU thresholding (>130 HU) to isolate candidate calcification regions  
  - Anatomical priors via TotalSegmentator for aorta, chambers  
  - Nearest‐neighbor upsampling + convolution to mitigate checkerboard artifacts  
- **Data Augmentation**  
  - 2D slice sampling focused on plaque‐containing slices  
  - 3D volume sampling (96³ cubes) with positive:negative ratio 4:1  
- **Training & Evaluation**  
  - Mixed‑precision training (AdamW, LR=1×10⁻⁴, up to 1600 epochs, early stopping)  
  - Metrics: Dice score, absolute error in Agatston & plaque volume :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}  
  - Per‑patient and per‑artery analyses
 
| Input Dim. | Model     | Dice ↑ | Agatston Error (mean ± std) ↓ |
|------------|-----------|--------|-------------------------------|
| **2D**     | RaUNet2D  | 0.56   | 21.04 ± 52.95                 |
|            | UNet2D    | 0.51   | 166.98 ± 248.76               |
|            | UNETR2D   | 0.41   | 229.38 ± 319.95               |
| **3D**     | RaUNet3D  | 0.87   | 52.14 ± 120.36                |
|            | UNet3D    | 0.84   | 45.49 ± 60.46                 |
