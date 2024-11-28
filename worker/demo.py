import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
def load_model(model_path, device):
    """Load a PyTorch model from a .pth file."""
    model = smp.Unet(
            encoder_name="resnet34",        # Encoder backbone
            encoder_weights=None,    # Pre-trained on ImageNet
            in_channels=1,  # Input channels (e.g., 1 for grayscale)
            classes=3  # Output classes (e.g., 3 for segmentation classes)
        )

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, transform):
    """
    Preprocess an image (JPG/PNG) into a NumPy array and apply Albumentations.

    Args:
        image_path (str): Path to the input image (JPG/PNG).
        transform (albumentations.Compose): Albumentations transform pipeline.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        # Load image and convert to grayscale
        img = Image.open(image_path).convert("L")  # Grayscale
        img_data = np.array(img)  # Convert to NumPy array

        print(f"Loaded image with shape: {img_data.shape}")

        # Apply Albumentations transforms
        augmented = transform(image=img_data)
        img_transformed = augmented["image"]  # Tensor with shape [C, H, W]

        print(f"Transformed image shape: {img_transformed.shape}")
        return img_transformed

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def overlay_image_mask(image, mask, output_path):
    """
    Overlay the image with the mask and save it.

    Args:
        image (numpy.ndarray): Input image (grayscale) with shape [H, W].
        mask (numpy.ndarray): Predicted mask with shape [H, W].
        output_path (str): Path to save the overlay.
    """
    try:
        # Convert image and mask to 2D arrays
        image = image.squeeze()  # Shape becomes [H, W]
        mask = mask.squeeze()  # Shape becomes [H, W]

        # Plot and save the overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")  # Grayscale image
        plt.imshow(mask, cmap="jet", alpha=0.5)  # Overlay mask
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Overlay saved at {output_path}")
    except Exception as e:
        print(f"Error during overlay generation: {e}")
        raise


def predict_segmentation(image_name):
    """
    Predict segmentation for a given image, and save the overlay image.

    Args:
        image_name (str): Name of the input image file (JPG/PNG).
    """
    model_path = "worker/best_UNet.pth"
    image_path = "worker/" + image_name
    output_path = "worker/output/" + image_name

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)

    # Define Albumentations transform pipeline
    transform = A.Compose([
        A.Resize(256, 256),  # Resize to 256x256
        A.Normalize(mean=0.5, std=0.5),  # Normalize to [-1, 1]
        ToTensorV2()  # Convert to PyTorch Tensor
    ])

    # Preprocess image
    preprocessed_image = preprocess_image(image_path, transform)

    # Add batch dimension and move to device
    preprocessed_image = preprocessed_image.unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    # Generate mask using the model
    with torch.no_grad():
        mask = model(preprocessed_image)  # Output shape: [1, C, H, W]
        mask = mask.argmax(dim=1).cpu().numpy()  # Convert to NumPy, shape: [1, H, W]

    # Save overlay
    overlay_image_mask(preprocessed_image.squeeze(0).cpu().numpy(), mask[0], output_path)


# Example usage
if __name__ == "__main__":
    # Example paths (replace with actual paths)
    image_name = "image_001113.jpg"  # Replace with your input image
    predict_segmentation(image_name)
