"""
Evaluate a pre-trained model on a test dataset.
Calculates metrics for the test dataset, including:
- Dice Score (Tumor+Kidney and Tumor only)
- Intersection over Union (IoU)
- Accuracy
- Recall

Outputs:
- Overall metrics for the test dataset.
- Per-case metrics.
"""

import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from monai.networks.nets import UNet, UNETR
from utils import prepare_datasets  # Replace with your dataset preparation function
import matplotlib.pyplot as plt


DEFAULT_HU_MIN = -512          # Minimum Hounsfield Unit (HU) for visualization
DEFAULT_HU_MAX = 512           # Maximum Hounsfield Unit (HU) for visualization
DEFAULT_KIDNEY_COLOR = [255, 0, 0]  # RGB color for kidneys (red)
DEFAULT_TUMOR_COLOR = [0, 0, 255]   # RGB color for tumors (blue)
DEFAULT_OVERLAY_ALPHA = 0.3    # Alpha (transparency) value for overlay
DEFAULT_PLANE = "axial"


def visualize_predictions(
    case_id,
    destination,
    predicted_segmentation,
    hu_min=DEFAULT_HU_MIN,
    hu_max=DEFAULT_HU_MAX,
    k_color=DEFAULT_KIDNEY_COLOR,
    t_color=DEFAULT_TUMOR_COLOR,
    alpha=DEFAULT_OVERLAY_ALPHA,
    plane=DEFAULT_PLANE,
):
    """
    Visualize ground truth and predicted segmentations as overlays on the image volume.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "case_00000").
    destination : str
        Output directory to save visualization images.
    predicted_segmentation : np.ndarray
        Predicted segmentation mask (shape: [D, H, W]).
    hu_min : int, optional
        Minimum Hounsfield Unit for grayscale scaling (default: -512).
    hu_max : int, optional
        Maximum Hounsfield Unit for grayscale scaling (default: 512).
    k_color : list of int, optional
        RGB color for kidney segmentation (default: [255, 0, 0]).
    t_color : list of int, optional
        RGB color for tumor segmentation (default: [0, 0, 255]).
    alpha : float, optional
        Opacity of the segmentation overlay (default: 0.3).
    plane : str, optional
        Visualization plane: "axial", "coronal", or "sagittal" (default: "axial").
    """
    plane = plane.lower()
    plane_opts = ["axial", "coronal", "sagittal"]
    if plane not in plane_opts:
        raise ValueError(f"Plane \"{plane}\" not understood. Must be one of {plane_opts}")

    # Prepare output location
    out_path = Path(destination)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load ground truth and volume
    vol, seg = load_case(case_id)
    vol = vol.get_data()
    seg = seg.get_data()
    seg = seg.astype(np.int32)

    # Use predicted segmentation instead of ground truth
    pred_seg = predicted_segmentation.astype(np.int32)

    # Convert to grayscale and segmentation colors
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max).astype(np.uint8)
    gt_seg_ims = class_to_color(seg, k_color, t_color).astype(np.uint8)
    pred_seg_ims = class_to_color(pred_seg, k_color, t_color).astype(np.uint8)

    # Overlay the segmentation
    gt_overlay = overlay(vol_ims, gt_seg_ims, seg, alpha)
    pred_overlay = overlay(vol_ims, pred_seg_ims, pred_seg, alpha)

    # Save overlay images
    for i in range(vol.shape[0]):
        gt_fpath = out_path / f"gt_{i:05d}.png"
        pred_fpath = out_path / f"pred_{i:05d}.png"
        imwrite(str(gt_fpath), gt_overlay[i])
        imwrite(str(pred_fpath), pred_overlay[i])

    print(f"Visualizations saved to {destination}")





def calculate_metrics(predictions, labels):
    """
    Calculate evaluation metrics between predictions and ground truth labels.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted segmentation mask (binary or multi-class).
    labels : np.ndarray
        Ground truth segmentation mask.

    Returns
    -------
    dict
        Dictionary containing:
        - "tk_dice": Tumor+Kidney Dice score.
        - "tu_dice": Tumor Dice score.
        - "iou": Intersection over Union (IoU).
        - "accuracy": Pixel accuracy.
        - "recall": Recall.
    """
    epsilon = 1e-6  # To avoid division by zero

    # Flatten predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Tumor+Kidney (tk) Dice
    tk_pred = predictions > 0  # Tumor+Kidney is any class > 0
    tk_gt = labels > 0
    tk_tp = np.sum(tk_pred & tk_gt)
    tk_fp = np.sum(tk_pred & ~tk_gt)
    tk_fn = np.sum(~tk_pred & tk_gt)
    tk_dice = (2 * tk_tp) / (2 * tk_tp + tk_fp + tk_fn + epsilon)

    # Tumor-only (tu) Dice
    tu_pred = predictions == 2  # Tumor is typically class 2
    tu_gt = labels == 2
    tu_tp = np.sum(tu_pred & tu_gt)
    tu_fp = np.sum(tu_pred & ~tu_gt)
    tu_fn = np.sum(~tu_pred & tu_gt)
    tu_dice = (2 * tu_tp) / (2 * tu_tp + tu_fp + tu_fn + epsilon)

    # Overall IoU, Accuracy, Recall
    tp = np.sum(predictions == labels)  # True positives for IoU
    fp = np.sum((predictions != labels) & (predictions > 0))
    fn = np.sum((predictions != labels) & (labels > 0))
    tn = np.sum((predictions == labels) & (predictions == 0))

    iou = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    recall = tp / (tp + fn + epsilon)

    return {
        "tk_dice": tk_dice,
        "tu_dice": tu_dice,
        "iou": iou,
        "accuracy": accuracy,
        "recall": recall,
    }


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on a test dataset and compute overall metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained PyTorch model.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    device : str
        Device to run the model on ('cuda' or 'cpu').

    Returns
    -------
    dict
        Dictionary containing overall metrics and per-case metrics.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    overall_metrics = {"tk_dice": [], "tu_dice": [], "iou": [], "accuracy": [], "recall": []}
    per_case_metrics = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            batch_size = images.size(0)

            for sample_idx in range(batch_size):
                # Prepare input
                image = images[sample_idx].unsqueeze(0).to(device)  # Add batch dim
                label = labels[sample_idx].cpu().numpy()  # Ground truth

                # Run the model
                output = model(image)
                prediction = output.argmax(dim=1).cpu().numpy()[0]  # Remove batch dim

                # Calculate metrics
                metrics = calculate_metrics(prediction, label)
                overall_metrics["tk_dice"].append(metrics["tk_dice"])
                overall_metrics["tu_dice"].append(metrics["tu_dice"])
                overall_metrics["iou"].append(metrics["iou"])
                overall_metrics["accuracy"].append(metrics["accuracy"])
                overall_metrics["recall"].append(metrics["recall"])

                # Store per-case metrics
                print(f"\nBatch {batch_idx}, Sample {sample_idx}")
                print(str(test_loader.dataset.cases[batch_idx]["image"]))
                case_id = test_loader.dataset.cases[batch_idx]["image"].split("\\")[-2]
                per_case_metrics.append(
                    {
                        "case_id": case_id,
                        "tk_dice": metrics["tk_dice"],
                        "tu_dice": metrics["tu_dice"],
                        "iou": metrics["iou"],
                        "accuracy": metrics["accuracy"],
                        "recall": metrics["recall"],
                    }
                )

                print(
                    f"Case {case_id} - Tumor+Kidney Dice: {metrics['tk_dice']:.4f}, Tumor Dice: {metrics['tu_dice']:.4f}, "
                    f"IoU: {metrics['iou']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}"
                )

    # Calculate overall averages
    overall_results = {
        "average_tk_dice": np.mean(overall_metrics["tk_dice"]),
        "average_tu_dice": np.mean(overall_metrics["tu_dice"]),
        "average_iou": np.mean(overall_metrics["iou"]),
        "average_accuracy": np.mean(overall_metrics["accuracy"]),
        "average_recall": np.mean(overall_metrics["recall"]),
        "per_case_metrics": per_case_metrics,
    }

    return overall_results

import numpy as np
import matplotlib.pyplot as plt

def inspect_ground_truth_matrix(label, slice_idx=None):
    """
    Inspect the ground truth segmentation matrix of a specific slice.

    Parameters
    ----------
    label : np.ndarray
        Ground truth segmentation mask of shape [D, H, W].
    slice_idx : int, optional
        Slice index to inspect. Defaults to the middle slice if not provided.
    """
    # Remove singleton dimensions
    label = np.squeeze(label)

    # Get the slice index (default to the middle slice if not provided)
    if slice_idx is None:
        slice_idx = label.shape[0] // 2  # Middle slice

    # Ensure slice_idx is valid
    slice_idx = min(slice_idx, label.shape[0] - 1)

    # Extract the slice
    label_slice = label[slice_idx, :, :]

    # Print the slice matrix
    print(f"Ground truth matrix for slice {slice_idx}:\n")
    print(label_slice)

    # Optionally visualize the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(label_slice, cmap="viridis")  # Use colormap for better visualization
    plt.colorbar(label="Class Label")
    plt.title(f"Ground Truth - Slice {slice_idx}")
    plt.axis("off")
    plt.show()



import os
from pathlib import Path

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelin = "unet"
    data_dir = "filtered_data"
    model_path = "results/unet_multiclass.pth"
    output_dir = "visualization_outputs"

    # Prepare datasets and loaders
    _, _, test_dataset, _, _, test_loader = prepare_datasets(data_dir)

    # Load the model
    if modelin == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif modelin == "unetr":
        model = UNETR(
            in_channels=1,
            out_channels=3,
            img_size=(128, 128, 128),
        )
        model_path = "results/unetr_multiclass.pth"

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Evaluate model
    import torch.nn.functional as F  # For softmax

    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            # Get raw model outputs
            outputs = model(images)  # Shape: [B, C, D, H, W]
            print(f"Raw outputs shape: {outputs.shape}")
            print(f"Raw outputs min: {outputs.min().item()}, max: {outputs.max().item()}")

            # Apply softmax along the channel dimension
            probabilities = F.softmax(outputs, dim=1)  # Shape: [B, C, D, H, W]
            print(f"Softmax probabilities shape: {probabilities.shape}")
            print(f"Probabilities min: {probabilities.min().item()}, max: {probabilities.max().item()}")

            # Convert probabilities to predictions (class labels)
            predictions = probabilities.argmax(dim=1).cpu().numpy()  # Shape: [B, D, H, W]

            # Convert labels to numpy for comparison
            labels = labels.cpu().numpy()
            images = images.cpu().numpy()

            # Visualize predictions for the first sample in the batch
            for sample_idx in range(images.shape[0]):
                case_path = test_loader.dataset.cases[batch_idx * images.shape[0] + sample_idx]["image"]
                case_id = Path(case_path).parent.name  # Use pathlib to get the folder name

                # Visualize predictions
                visualize_predictions(
                    case_id,
                    destination=os.path.join(output_dir, case_id),
                    predicted_segmentation=predictions[sample_idx],
                )

            # Stop after one batch for debugging purposes
            break
