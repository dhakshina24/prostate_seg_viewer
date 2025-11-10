"""Functions for loading and normalizing MRI and segmentation data"""
# Import necessary libraries
import SimpleITK as sitk
import numpy as np


def normalize_mri(img):
    """Normalize and scale MRI image slice to [0, 255] for visualization."""
    img = img.astype(np.float32)
    # Clip 1st and 99th percentiles
    low, high = np.percentile(img, (1, 99)) 
    img = np.clip(img, low, high)
    # Min-Max Normalization
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)
    return img

def load_mri(img_path):
    """Load a 3D MRI image, normalize it, and return it as a NumPy array (z, y, x)."""
    img = sitk.ReadImage(img_path)
    meta_data = load_metadata(img)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = normalize_mri(img_arr)
    return meta_data, img_arr

def load_metadata(img):
    """Extract metadata such as size, spacing, origin and orientation"""
    size = img.GetSize()
    spacing = tuple(round(s, 2) for s in img.GetSpacing())
    origin = tuple(round(o, 2) for o in img.GetOrigin())
    direction =  np.array(img.GetDirection()).reshape(3, 3)

    orientation = "Axial"
    if direction[2, 2] < 0.5:
        orientation = "Sagittal" if direction[0, 2] > 0.5 else "Coronal"

    data = {
        "Dimensions" : f"{size[0]} x {size[1]} x {size[2]}", 
        "Spacing (mm)" : f"{spacing[0]} x {spacing[1]} x {spacing[2]}",
        "Origin (mm)" : origin, 
        "Direction" : orientation, 
    }
    return data

def load_mask(mask_path):
    """Load the mask and return it as a NumPy array (z, y, x)."""
    mask = sitk.ReadImage(mask_path)
    mask_data = load_metadata(mask)
    mask_arr = sitk.GetArrayFromImage(mask)
    return mask_data, mask_arr