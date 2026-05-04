## 📌 Project Overview
This repository presents a comprehensive computer vision pipeline for detecting and localizing pneumonia from chest X-ray images. The project bridges the gap between high-level classification and pixel-level segmentation, providing a robust tool for medical image analysis.

As a dual-degree technology student, I developed this project to compare state-of-the-art architectures and evaluate their performance in clinical diagnostics.

## 🏗️ Architecture & Methodology
The project is divided into two distinct phases:

### 1. Classification (CNN vs. ResNet18)
*   **Objective**: Rapid screening to distinguish between Normal and Pneumonia cases.
*   **Models**: Custom CNN vs. Pre-trained ResNet18 (Transfer Learning).
*   **Result**: Achieved an impressive **97% accuracy** on the validation set.

### 2. Segmentation (U-Net)
*   **Objective**: Precise localization of pulmonary opacities.
*   **Model**: **U-Net** architecture using PyTorch and OpenCV.
*   **Implementation**: Utilizes deep learning layers to generate masks pinpointing affected areas.



## 📊 Key Results
*   **Classification Accuracy**: 97%
*   **Segmentation**: Pixel-perfect masking of infection zones using U-Net.
*   **Tools**: Built with a tech stack including Python, PyTorch, and OpenCV.

## 📁 Repository Structure
*   `main.py`: Entry point for classification and training logic.
*   `unet_segmentation.py`: Implementation of the U-Net architecture and segmentation pipeline.
*   `.gitignore`: Optimized to exclude heavy datasets while keeping the repo lightweight.

## 🔗 Data & Models Access
Due to GitHub's storage limitations, the raw dataset and trained model weights (`.pth`) are hosted externally.

> 📥 **[Download Dataset (archive/) & Pre-trained Models Here](//)**

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Nadabenamar/Pneumonia-Detection-and-Segmentation-A-Comparative-Study-of-CNN-ResNet18-and-U-Net.git](https://github.com/Nadabenamar/Pneumonia-Detection-and-Segmentation-A-Comparative-Study-of-CNN-ResNet18-and-U-Net.git)
## Project Structure
The project is organized as follows:
*   `main.py`: Core logic for Pneumonia classification (CNN & ResNet18).
*   `unet_segmentation.py`: Implementation of the U-Net architecture for lung mask generation.
*   `archive/`: Contains the Kaggle Chest X-Ray dataset (organized into train/val/test).
*   `segmentation/`: Includes expert-labeled images and masks for the U-Net module.
  ## 📁 Model Weights
To keep the repository lightweight, trained model weights (`.pth` files) are not included. 
You can generate them by:
1. Running the training scripts provided in `src/`.
2. Or downloading our pre-trained weights from https://drive.google.com/file/d/1ZORIlI-Ze1byNV3DOpBNh2B3u7Dzb4PO/view?usp=drive_link .

Place the downloaded `model.pth` in the `models/` directory before running inference.
