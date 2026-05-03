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
