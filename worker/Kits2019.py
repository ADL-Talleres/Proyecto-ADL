import os
import torch
import nibabel as nib
import numpy as np
from monai.data import Dataset

class Kits2019Dataset(Dataset):
    """
    A custom PyTorch Dataset for loading and processing 3D medical scans 
    from the KiTS19 dataset. This dataset is designed to load NIfTI (.nii.gz) 
    images and corresponding labels (segmentation masks).

    Parameters
    ----------
    cases : list of dict
        A list of dictionaries, where each dictionary contains the file paths 
        for the image and label NIfTI files. For example:
        [{"image": "path/to/image.nii.gz", "label": "path/to/label.nii.gz"}, ...].
    transform : callable, optional
        A MONAI transform pipeline to be applied to the image and label. 
        Default is None.

    Attributes
    ----------
    cases : list of dict
        Stores the list of file paths for the dataset.
    transform : callable or None
        Stores the transform pipeline for preprocessing.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Loads the image and label for the given index, applies the transform 
        if provided, and returns the processed tensors.

    Examples
    --------
    >>> cases = [
    ...     {"image": "case_00000/imaging.nii.gz", "label": "case_00000/segmentation.nii.gz"},
    ...     {"image": "case_00001/imaging.nii.gz", "label": "case_00001/segmentation.nii.gz"}
    ... ]
    >>> dataset = Kits2019Dataset(cases)
    >>> len(dataset)
    2
    >>> image, label = dataset[0]
    >>> print(image.shape, label.shape)
    torch.Size([1, D, H, W]) torch.Size([1, D, H, W])
    """

    def __init__(self, cases, transform=None):
        """
        Initializes the dataset with file paths and an optional transform.

        Parameters
        ----------
        cases : list of dict
            A list of dictionaries containing "image" and "label" paths.
        transform : callable, optional
            A MONAI transform pipeline to apply to the samples.
        """
        self.cases = cases
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.cases)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing the preprocessed image tensor and label tensor. 
            Each tensor has a shape of [1, D, H, W], where D is depth, H is height, 
            and W is width.
        """
        # Load paths for image and label
        sample_paths = self.cases[idx]

        # Load `.nii.gz` files
        image = nib.load(sample_paths["image"]).get_fdata(dtype=np.float32)
        label = nib.load(sample_paths["label"]).get_fdata(dtype=np.float32)

        # Convert to PyTorch tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).unsqueeze(0)  # Add channel dimension

        print(image.shape)  # Verify the shape
        print(label.shape)  # Verify the shape

        # Create sample dictionary
        sample = {"image": image, "label": label}

        # Apply transforms, if any
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]
