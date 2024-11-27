import torch
import os
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    Spacingd,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
)
from monai.data import MetaTensor
from matplotlib import pyplot as plt
from monai.networks.nets import UNet


def load_model(model_path, device):
    """Load a PyTorch model from a .pth file."""
    model = UNet(
    spatial_dims=3,  # Correct value for 3D convolutions
    in_channels=1,   # Match input data channels
    out_channels=3,  # Number of segmentation classes
    channels=(16, 32, 64, 128, 256),  # Feature map sizes for each level
    strides=(2, 2, 2, 2),  # Downsampling factors for each level
    ).to(device)

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, transform):
    """
    Preprocess a JPG image using the specified MONAI transforms.

    Args:
        image_path (str): Path to the JPG image.
        transform (Compose): MONAI transform pipeline.

    Returns:
        MetaTensor: Preprocessed image.
    """
    try:
        # Load the JPG image
        img = Image.open(image_path).convert("L")  # Convert to grayscale if needed
        img_data = np.array(img)
        
        # Create a 3D array with 30 identical slices
        img_3d = np.tile(img_data, (30, 1, 1))  # Shape -> (30, H, W)

        # Save as a NIfTI image (if needed for intermediate use)
        affine = np.eye(4)  # Identity affine matrix
        nifti_img = nib.Nifti1Image(img_3d, affine)
        nib.save(nifti_img, "output_image.nii.gz")  # Optional: Save for debugging
        
        print(f"Generated NIfTI with shape: {img_3d.shape}")

        # Convert to MetaTensor (add channel dimension)
        img_meta = MetaTensor(img_3d[None, ...])  # Add channel dimension -> shape (1, 30, H, W)

        # Apply the transform
        transformed = transform({"image": img_meta})
        print(f"Transformed shape: {transformed['image'].shape}")
        
        return transformed["image"]

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise



def overlay_image_mask(image, mask, output_path):
    """
    Overlay the image with the mask and save it.

    Args:
        image (numpy.ndarray): The input image (shape: [D, H, W]).
        mask (numpy.ndarray): The predicted mask (shape: [C, D, H, W] or [D, H, W]).
        output_path (str): Path to save the overlay.
    """
    # Remove batch/channel dimension from image if present
    image = image.squeeze()  # Shape becomes [D, H, W]
    mask = mask.squeeze()  # Shape becomes [C, D, H, W] or [D, H, W]

    # If mask has multiple channels (e.g., [C, D, H, W]), take argmax along channel axis
    if len(mask.shape) == 4:
        mask = np.argmax(mask, axis=0)  # Shape becomes [D, H, W]

    # Select the middle slice along the depth dimension
    mid_idx = image.shape[0] // 2
    image_slice = image[mid_idx, :, :]  # Shape: [H, W]
    mask_slice = mask[mid_idx, :, :]  # Shape: [H, W]

    # Plot and save the overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image_slice, cmap="gray")  # Grayscale image
    plt.imshow(mask_slice, cmap="jet", alpha=0.5)  # Overlay mask
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()




def predict_segmentation(image_name):
    """
    Predict segmentation for a given image using a specified model,
    and save the overlay image to the output path.

    Args:
        model_path (str): Path to the model (.pth file).
        image_path (str): Path to the input image (.nii.gz file).
        output_path (str): Path to save the overlay image.
    """
    model_path = "../worker/unet_multiclass.pth"
    image_path = "../back/upload/" + image_name
    output_path = "../back/uploads_reason/" + image_name.split(".")[0] + ".jpg"
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)

    # Define transforms
    transform = Compose([
        Resized(keys=["image"], spatial_size=(256, 256, 256)),
        ScaleIntensityd(keys="image"),
        EnsureTyped(keys=["image"]),
    ])

    # Preprocess image
    preprocessed_image = preprocess_image(image_path, transform)

    # Move to device
    preprocessed_image = preprocessed_image.to(device)

    # Generate mask using the model
    with torch.no_grad():
        mask = model(preprocessed_image[None, ...])  # Add batch dimension
    mask = mask.cpu().numpy()  # Convert to numpy

    # Save overlay
    overlay_image_mask(preprocessed_image.cpu().numpy(), mask, output_path)
    print(f"Overlay saved to {output_path}")

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image   

def test_overlay_logic(image_path, groundtruth_path, output_path=None):
    """
    Test overlay logic using an image and its ground truth mask.

    Args:
        image_path (str): Path to the input image (.nii or .nii.gz).
        groundtruth_path (str): Path to the ground truth mask (.nii or .nii.gz).
        output_path (str): Path to save the overlay visualization (optional).
    """
    # Load the image and ground truth mask
    image = nib.load(image_path).get_fdata()
    groundtruth = nib.load(groundtruth_path).get_fdata()

    # Check if the shapes match
    if image.shape != groundtruth.shape:
        raise ValueError(f"Image shape {image.shape} and ground truth shape {groundtruth.shape} do not match.")

    print(f"Loaded image shape: {image.shape}")
    print(f"Loaded ground truth shape: {groundtruth.shape}")

    # Select the middle slice along the depth (z-axis)
    mid_idx = image.shape[0] // 2
    image_slice = image[mid_idx, :, :]  # Shape: [H, W]
    groundtruth_slice = groundtruth[mid_idx, :, :]  # Shape: [H, W]

    # Plot the overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image_slice, cmap="gray")  # Display the image
    plt.imshow(groundtruth_slice, cmap="jet", alpha=0.5)  # Overlay the ground truth mask
    plt.axis("off")

    # Save or show the result
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"Overlay visualization saved to {output_path}")
    else:
        plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example paths (replace with actual paths)

    image_name = "image_000037.jpg"
    predict_segmentation(image_name)


