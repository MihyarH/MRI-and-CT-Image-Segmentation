# preprocessing.py
import torchio as tio
import numpy as np

from dataLoader import load_nifti_file, visualize_slice


def preprocess_image(image, label, target_shape=(128, 128, 128)):
    """Preprocess the image and label with normalization and resizing."""
    # Convert to torchio Image objects
    image = tio.ScalarImage(tensor=image[np.newaxis, ...])  # Adding channel dimension
    label = tio.LabelMap(tensor=label[np.newaxis, ...])  # Adding channel dimension

    # Create a subject containing the image and label
    subject = tio.Subject(
        image=image,
        label=label
    )

    # Define transforms
    transform = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1), keys=['image']),  # Normalize intensity for image
        tio.Resize(target_shape)  # Resize to target shape
    ])

    # Apply the transforms to the subject
    transformed_subject = transform(subject)

    # Extract transformed image and label as tensors
    preprocessed_image = transformed_subject.image.data
    preprocessed_label = transformed_subject.label.data

    return preprocessed_image, preprocessed_label
