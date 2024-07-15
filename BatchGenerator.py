import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
import time

class BatchGenerator:
    def __init__(self, dataset, batch_size, augmentations=None, augment_factor=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.augment_factor = augment_factor

    def __len__(self):
        return len(self.dataset) * self.augment_factor

    def __iter__(self):
        batch_images = []
        batch_masks = []
        batch_shapes = []

        for idx in range(len(self.dataset)):
            for _ in range(self.augment_factor):
                image, mask, original_shape = self.dataset[idx]

                if self.augmentations:
                    image, mask = apply_transforms(image, mask, self.augmentations)

                batch_images.append(image)
                batch_masks.append(mask)
                batch_shapes.append(original_shape)

                if len(batch_images) == self.batch_size:
                    yield torch.stack(batch_images), torch.stack(batch_masks), batch_shapes
                    batch_images = []
                    batch_masks = []
                    batch_shapes = []

        if batch_images:
            yield torch.stack(batch_images), torch.stack(batch_masks), batch_shapes

    @staticmethod
    def find_optimal_batch_size(model, device, dataset, starting_batch_size=8, increment=8, max_memory_usage=0.9):
        batch_size = starting_batch_size
        if torch.cuda.is_available():
            memory_available = torch.cuda.get_device_properties(device).total_memory
            memory_limit = memory_available * max_memory_usage
        else:
            memory_limit = None

        while True:
            try:
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                images, masks, original_shapes = next(iter(loader))
                images, masks = images.to(device), masks.to(device)

                start_time = time.time()
                outputs = model(images)
                end_time = time.time()

                if memory_limit:
                    memory_used = torch.cuda.memory_allocated(device)
                    print(f"Batch size: {batch_size}, Memory used: {memory_used / (1024**2):.2f} MB, Time taken: {end_time - start_time:.4f} seconds")

                    if memory_used >= memory_limit:
                        break
                else:
                    print(f"Batch size: {batch_size}, Time taken: {end_time - start_time:.4f} seconds")

                batch_size += increment
            except RuntimeError as e:
                print(f"Error with batch size {batch_size}: {e}")
                break

        return batch_size - increment

def apply_transforms(image, mask, transforms):
    """
    Apply the same transformations to both image and mask.
    Args:
    - image (Tensor): The input image tensor.
    - mask (Tensor): The corresponding mask tensor.
    - transforms (torchvision.transforms.Compose): The composed transformations to apply.
    Returns:
    - transformed_image (Tensor): The transformed image tensor.
    - transformed_mask (Tensor): The transformed mask tensor.
    """
    image_pil = to_pil_image(image)
    mask_pil = to_pil_image(mask)

    # Apply transformations to both image and mask
    image_transformed = transforms(image_pil)
    mask_transformed = transforms(mask_pil)

    # Convert back to tensors
    image_transformed = to_tensor(image_transformed)
    mask_transformed = to_tensor(mask_transformed)

    return image_transformed, mask_transformed
