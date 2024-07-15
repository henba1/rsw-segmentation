import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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
        image = normalize_and_pad_image(self.images[idx], self.target_size, self.resize_transform)
        coordinates = points_to_coordinates(self.mask_points[idx])

        # Create an empty mask if coordinates are empty
        if coordinates.size == 0:
            mask = torch.zeros((1, *self.target_size), dtype=torch.float32)
        else:
            mask = create_and_pad_mask(self.images[idx].shape, coordinates, self.target_size, self.resize_transform)

        return image, mask, image.shape

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
    return padded_image

def normalize_and_pad_image(image, target_size=(512, 512), resize_transform=None):
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    padded_image = pad_to_target_size((image * 255).astype(np.uint8), target_size)  # Pad image
    resized_image = resize_transform(padded_image)  # Resize image to target size
    return transforms.ToTensor()(resized_image)  # Convert to tensor

def create_and_pad_mask(image_shape, coordinates, target_size=(512, 512), resize_transform=None):
    mask = np.zeros(image_shape, dtype=np.float32)
    if coordinates.size > 0:
        rr, cc = polygon(coordinates[:, 1], coordinates[:, 0], shape=image_shape)
        mask[rr, cc] = 1.0
    padded_mask = pad_to_target_size((mask * 255).astype(np.uint8), target_size)
    resized_mask = resize_transform(padded_mask) 
    return transforms.ToTensor()(resized_mask) 

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

def visual_inspect(image_tensor, mask_tensor, save_path=None):
    #for visual inspection of resulting images
    def overlay_mask_on_image(image, mask, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        ax.imshow(mask, cmap='jet', alpha=0.4)  # Use alpha to make mask semi-transparent
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    image_np = image_tensor.squeeze().numpy() * 0.5 + 0.5 
    mask_np = mask_tensor.squeeze().numpy()
    overlay_mask_on_image(image_np, mask_np, save_path)
