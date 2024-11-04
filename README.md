# MRI and CT Image Segmentation with 3D U-Net

This project implements a 3D U-Net model to perform segmentation on MRI and CT images, specifically using the AMOS dataset. The model is designed to identify and delineate regions in medical images through voxel-level binary segmentation. The training pipeline is optimized to run on a GPU and includes real-time progress tracking and accuracy metrics.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Dependencies](#dependencies)
- [Credits](#credits)

## Project Overview
This project uses a 3D U-Net architecture, a popular choice for volumetric data segmentation, ideal for medical imaging tasks. The AMOS dataset, which contains CT and MRI scans, serves as the data source. Each scan is preprocessed (resized, normalized) and used to train the model to predict segmentation masks.

### Model Architecture
The U-Net model employs an encoder-decoder structure:
- **Encoder Path**: Downsamples and extracts features.
- **Decoder Path**: Upsamples and combines encoder features to reconstruct segmentation maps.
- **Output**: Produces voxel-level segmentation for each input scan.

### Performance Metrics
- **Loss**: Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss)
- **Dice Score**: Evaluates overlap between predicted and ground truth masks, a common metric for segmentation tasks.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/mri-ct-segmentation.git
   cd mri-ct-segmentation

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
