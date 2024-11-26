import torch
import numpy as np
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, EnsureTyped
from monai.data import DataLoader
from Kits2019 import Kits2019Dataset
from collections import OrderedDict
import nibabel as nib

# Arguments
args = {
    "num_classes": 3,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "checkpoint_path": "./unet_multiclass.pth",
}

# Classes
clase = {
    0: "Background",
    1: "Kidney",
    2: "Tumor"
}

# Load model function
def load_model(num_classes, checkpoint_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(model_state_dict)
    return model.to(device).eval()

# Remove 'module.' prefix from state_dict
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # Remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

# Save segmentation mask
def save_segmentation(pred_mask, save_path):
    pred_nifti = nib.Nifti1Image(pred_mask.astype(np.uint8), np.eye(4))
    nib.save(pred_nifti, save_path)
    print(f"Segmentation saved to {save_path}")

# Main segmentation function
def get_segmentation(image_path, save_path=None):
    # Define transforms
    transform = Compose([
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        EnsureTyped(keys=["image"])
    ])
    
    # Prepare dataset and dataloader
    case = [{"image": image_path, "label": None}]
    single_dataset = Kits2019Dataset(case, transform=transform)
    single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)

    # Perform inference
    with torch.no_grad():
        for batch in single_loader:
            image = batch[0].to(args["device"])
            outputs = model(image)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()  # Generate segmentation mask
            
            if save_path:
                save_segmentation(preds, save_path)
            
            return preds

# Load model once during initialization
model = load_model(args["num_classes"], args["checkpoint_path"], args["device"])

# Example usage
if __name__ == "__main__":
    # Example usage for a backend service
    test_image_path = "./kits19/case_00000/filtered_imaging.nii.gz"
    output_path = "./case_00000_segmentation.nii.gz"
    mask = get_segmentation(test_image_path, save_path=output_path)
    print("Segmentation mask generated.")
