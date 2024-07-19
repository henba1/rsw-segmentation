import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.draw import polygon
import matplotlib.pyplot as plt


class DataProcessing(Dataset):
    def __init__(self, images, mask_points, original_size, target_size=(512, 512), resize_transform=None):
        """
        Args:
        - images (list): List of image arrays.
        - mask_points (list): List of flat coordinate points for each image.
        - target_size (tuple): Target size for resizing (height, width).
        - resize_transform (callable, optional): A torchvision transform for resizing images and masks.
        """
        self.images = images
        self.mask_points = mask_points
        self.original_size = original_size
        self.target_size = target_size
        self.resize_transform = resize_transform if resize_transform else transforms.Resize(target_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tensor = self.images[idx]
        image, padding = normalize_and_pad_image(self.images[idx], self.target_size, self.resize_transform)
        coordinates = points_to_coordinates(self.mask_points[idx])
        mask, _ = create_and_pad_mask(self.images[idx].shape, coordinates, self.target_size, self.resize_transform)
        # print('=================')
        # print(self.images[idx].shape)
        # print(self.original_size[idx])
    
        return image_tensor, image, mask, self.original_size[idx], padding  # Return padding information

    @staticmethod
    def unpad_and_resize(tensor, original_shape, padding):
        height, width = original_shape
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


def visual_inspect(image_tensor, mask_tensor, pred_tensor, resize_dim, save_path=None, original_shape=None, bin_thresh=0.5):
    def save_image(save_dir, filename):
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Convert tensors to numpy arrays and adjust intensity range
    image_np = image_tensor.squeeze().numpy() * 0.5 + 0.5
    mask_np = mask_tensor.squeeze().numpy()
    pred_np = pred_tensor.squeeze().numpy() if pred_tensor is not None else None

    #determine cut-off in width dim 
    image_np = image_np[:,:resize_dim]
    mask_np = mask_np[:,:resize_dim]
    pred_np = pred_np[:,: resize_dim]
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Initial Image
        plt.imshow(image_np, cmap='gray')
        save_image(save_path, "initial_image.png")

        # Ground Truth Overlay
        plt.imshow(image_np, cmap='gray')
        plt.imshow(mask_np, cmap='jet', alpha=0.4)
        save_image(save_path, "ground_truth_overlay.png")

        if pred_np is not None:
            # Prediction Overlay
            plt.imshow(image_np, cmap='gray')
            plt.imshow(pred_np, cmap='jet', alpha=0.4)
            save_image(save_path, "prediction_overlay.png")

            # Ground Truth and Prediction Overlay
            plt.imshow(image_np, cmap='gray')
            plt.imshow(mask_np, cmap='jet', alpha=0.4)
            plt.imshow(pred_np, cmap='jet', alpha=0.4)
            save_image(save_path, "gt_pred_overlay.png")

            # Binarized Prediction
            binarized_pred = (pred_np > bin_thresh).astype(np.float32)
            plt.imshow(image_np, cmap='gray')
            plt.imshow(binarized_pred, cmap='jet', alpha=0.5)
            save_image(save_path, f"binarized_prediction_thresh={bin_thresh}.png")
    else: 
        print("visual_inspect: Please provide a save_path to save the image.")
        return

