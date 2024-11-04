# dataLoader.py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nifti_file(file_path):
    """Load a .nii.gz file and return the numpy array data."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def visualize_slice(image, label, slice_idx):
    """Visualize a specific slice of the 3D image and its label."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[:, :, slice_idx], cmap="gray")
    ax[0].set_title("Image Slice")
    ax[1].imshow(label[:, :, slice_idx], cmap="gray")
    ax[1].set_title("Label Slice")
    plt.show()
