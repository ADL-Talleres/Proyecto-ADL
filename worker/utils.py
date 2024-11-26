"""
This module prepares datasets and data loaders for 3D medical imaging tasks using the MONAI framework. 
It supports applying transformations, splitting the dataset into training, validation, and test sets, 
and generating PyTorch DataLoaders.

Functions:
- prepare_datasets: Splits the dataset, applies transformations, and returns datasets and data loaders.
- get_transforms: Returns a transformation pipeline based on the purpose (train/test) and model type.
"""

import os
from monai.transforms import (
    Compose, Spacingd, ScaleIntensityd, RandFlipd, RandRotate90d, RandZoomd,
    EnsureTyped, ResizeWithPadOrCropd
)
from torch.utils.data import DataLoader
import random
from Kits2019 import Kits2019Dataset

def prepare_datasets(data_dir, split_ratios=(0.7, 0.2, 0.1), transform=None):
    """
    Prepares the datasets and data loaders for training, validation, and testing.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory. Each case folder should contain 
        "filtered_imaging.nii.gz" and "filtered_segmentation.nii.gz".
    split_ratios : tuple of float, optional
        Proportions to split the dataset into train, validation, and test sets. 
        Default is (0.7, 0.2, 0.1).
    transform : callable, optional
        Transformation pipeline to apply to the dataset. If None, a default pipeline 
        is used.

    Returns
    -------
    tuple
        A tuple containing:
        - train_dataset : monai.data.Dataset
        - val_dataset : monai.data.Dataset
        - test_dataset : monai.data.Dataset
        - train_loader : torch.utils.data.DataLoader
        - val_loader : torch.utils.data.DataLoader
        - test_loader : torch.utils.data.DataLoader
    """
    cases = []

    # Collect valid case folders
    for case in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, case)
        if os.path.isdir(case_dir) and case.startswith("case"):
            try:
                case_number = int(case.split("_")[-1])
                if case_number < 210:  # Filtering based on case number
                    cases.append({
                        "image": os.path.join(case_dir, "filtered_imaging.nii.gz"),
                        "label": os.path.join(case_dir, "filtered_segmentation.nii.gz"),
                    })
            except ValueError:
                continue

    # Shuffle and split the dataset
    random.shuffle(cases)
    num_cases = len(cases)
    train_end = int(split_ratios[0] * num_cases)
    val_end = train_end + int(split_ratios[1] * num_cases)

    train_cases = cases[:train_end]
    val_cases = cases[train_end:val_end]
    test_cases = cases[val_end:]

    # Use default transforms if none are provided
    transform = transform or get_transforms(purpose="train")

    # Create datasets and data loaders
    train_dataset = Kits2019Dataset(train_cases, transform=transform)
    val_dataset = Kits2019Dataset(val_cases, transform=transform)
    test_dataset = Kits2019Dataset(test_cases, transform=get_transforms(purpose="test"))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_transforms(purpose="test", model_name="sant"):
    """
    Returns a transformation pipeline for medical imaging data.

    Parameters
    ----------
    purpose : str, optional
        The purpose of the transformation pipeline. Options are:
        - "train": Includes augmentation for training.
        - "test": Basic preprocessing for validation or testing. 
        Default is "test".
    model_name : str, optional
        The model type for which the transformations are tailored. Options are:
        - "unet": Default target size is (256, 256, 256).
        - "unetr": Target size is (384, 384, 384).
        - Other: Default target size is (128, 128, 128).
        Default is "unet".

    Returns
    -------
    monai.transforms.Compose
        A MONAI Compose object containing the transformation pipeline.
    """
    if model_name == "unet":
        target_size = (256, 256, 256)
    elif model_name == "unetr":
        target_size = (384, 384, 384)
    else:
        target_size = (128, 128, 128)

    # Base preprocessing transforms
    base_transforms = [
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=target_size
        ),
        ScaleIntensityd(keys="image"),
        EnsureTyped(keys=["image", "label"])
    ]

    # Add augmentations for training
    if purpose == "train":
        return Compose(base_transforms + [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
        ])

    return Compose(base_transforms)
