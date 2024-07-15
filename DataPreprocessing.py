import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.draw import polygon
import matplotlib.pyplot as plt

class DataPreprocessing(Dataset):
    def __init__(self, images, mask_points, target_size=(512, 512), resize_transform=None):
        """
        Args:
        - images (list): List of image arrays.
        - mask_points (list): List of flat coordinate points for each image.
        - target_size (tuple): Target size for resizing (height, width).
        - resize_transform (callable, optional): A torchvision transform for resizing images and masks.
        """
        self.images = images
        self.mask_points = mask_points
        self.target_size = target_size
        self.resize_transform = resize_transform if resize_transform else transforms.Resize(target_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, padding = normalize_and_pad_image(self.images[idx], self.target_size, self.resize_transform)
        coordinates = points_to_coordinates(self.mask_points[idx])
        mask, _ = create_and_pad_mask(self.images[idx].shape, coordinates, self.target_size, self.resize_transform)
        return image, mask, self.images[idx].shape, padding  # Return padding information

    @staticmethod
    def unpad_and_resize(tensor, original_shape, padding):
        height, width = original_shape[:2]
        image_pil = to_pil_image(tensor)
        unpadded_image = transforms.functional.crop(image_pil, 0, 0, height, width)  # Remove padding
        resized_image = transforms.Resize((height, width))(unpadded_image)  # Resize image to original size
        return to_tensor(resized_image)

def normalize_image(image, resize_transform):
    image = image.astype(np.float32) / 255.0 
    image = (image - 0.5) / 0.5  
    image = Image.fromarray((image * 255).astype(np.uint8)) 
    image = resize_transform(image)  
    return transforms.ToTensor()(image)  

def create_mask(image_shape, coordinates, resize_transform):
    mask = np.zeros(image_shape, dtype=np.float32)
    rr, cc = polygon(coordinates[:, 1], coordinates[:, 0], shape=image_shape)
    mask[rr, cc] = 1.0
    mask = Image.fromarray((mask * 255).astype(np.uint8)) 
    mask = resize_transform(mask)  
    return transforms.ToTensor()(mask) 

def pad_to_target_size(image, target_size=(512, 512)):
    height, width = image.shape[:2]
    pad_height = max(0, target_size[0] - height)
    pad_width = max(0, target_size[1] - width)
    
    padding = (0, 0, pad_width, pad_height)  # (left, top, right, bottom)
    padded_image = transforms.functional.pad(Image.fromarray(image), padding)
    return padded_image, padding

def normalize_and_pad_image(image, target_size=(512, 512), resize_transform=None):
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    padded_image, padding = pad_to_target_size((image * 255).astype(np.uint8), target_size)  # Pad image
    resized_image = resize_transform(padded_image)  # Resize image to target size
    return transforms.ToTensor()(resized_image), padding  # Convert to tensor

def create_and_pad_mask(image_shape, coordinates, target_size=(512, 512), resize_transform=None):
    mask = np.zeros(image_shape, dtype=np.float32)
    if coordinates.size > 0:
        rr, cc = polygon(coordinates[:, 1], coordinates[:, 0], shape=image_shape)
        mask[rr, cc] = 1.0
    padded_mask, padding = pad_to_target_size((mask * 255).astype(np.uint8), target_size)
    resized_mask = resize_transform(padded_mask) 
    return transforms.ToTensor()(resized_mask), padding 

def points_to_coordinates(points):
    """
    Converts a flat list of points into a 2D NumPy array of coordinates.
    Args:
    - points (list): A flat list of points where two consecutive points are (x, y) coordinates.
    Returns:
    - coordinates (np.array): A 2D NumPy array of shape (n, 2) where n is the number of points.
    """
    assert len(points) % 2 == 0, "The number of points must be even."
    coordinates = np.array(points).reshape(-1, 2)
    return coordinates

def visual_inspect(image_tensor, mask_tensor, pred_tensor=None, save_path=None):
    def overlay_mask_on_image(image, mask, pred=None, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 1 row, 3 columns
        axes[0].imshow(image, cmap='gray')
        axes[0].imshow(mask, cmap='jet', alpha=0.4)
        axes[0].set_title("Ground Truth Overlay")
        axes[0].axis('off')

        if pred is not None:
            axes[1].imshow(image, cmap='gray')
            axes[1].imshow(pred, cmap='jet', alpha=0.4)
            axes[1].set_title("Prediction Overlay")
            axes[1].axis('off')
            
            axes[2].imshow(image, cmap='gray')
            axes[2].imshow(mask, cmap='jet', alpha=0.4)
            axes[2].imshow(pred, cmap='jet', alpha=0.4)
            axes[2].set_title("Ground Truth and Prediction Overlay")
            axes[2].axis('off')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    image_np = image_tensor.squeeze().numpy() * 0.5 + 0.5
    mask_np = mask_tensor.squeeze().numpy()
    pred_np = pred_tensor.squeeze().numpy() if pred_tensor is not None else None

    overlay_mask_on_image(image_np, mask_np, pred=pred_np, save_path=save_path)
