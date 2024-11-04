# dataset.py
import os
import torchio as tio
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_shape=(128, 128, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_shape = target_shape
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

        # Define individual transforms for images and labels
        self.image_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.Resize(target_shape)
        ])
        self.label_transform = tio.Resize(target_shape)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label paths
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load data
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Convert to TorchIO ScalarImage and LabelMap and add channel dimensions
        image = tio.ScalarImage(tensor=image[np.newaxis, ...])  # Add channel dimension
        label = tio.LabelMap(tensor=label[np.newaxis, ...])     # Add channel dimension

        # Apply transformations independently to ensure consistent shape
        transformed_image = self.image_transform(image)
        transformed_label = self.label_transform(label)

        # Extract tensors after transformation
        image_tensor = transformed_image.data
        label_tensor = transformed_label.data

        return image_tensor, label_tensor
