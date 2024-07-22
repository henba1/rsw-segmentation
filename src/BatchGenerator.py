import time
import torch
from torch import stack
from DataProcessing import apply_transforms

class BatchGenerator:
    def __init__(self, dataset, batch_size, augmentations=None, augment_factor=1, add_noise=False, noise_params=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.augment_factor = augment_factor
        self.add_noise = add_noise
        self.noise_params = noise_params

    def __len__(self):
        return len(self.dataset) * self.augment_factor

    def __iter__(self):
        batch_original_images = []
        batch_images = []
        batch_masks = []
        batch_shapes = []
        batch_paddings = []

        for idx in range(len(self.dataset)):

            original_image, image, mask, original_shape, padding = self.dataset[idx]
            # Include the original image
            batch_original_images.append(original_image)
            batch_images.append(image)
            batch_masks.append(mask)
            batch_shapes.append(original_shape)
            batch_paddings.append(padding)

            if len(batch_images) == self.batch_size:
                yield batch_original_images, stack(batch_images), stack(batch_masks), batch_shapes, batch_paddings
                batch_original_images = []
                batch_images = []
                batch_masks = []
                batch_shapes = []
                batch_paddings = []

            for _ in range(self.augment_factor - 1):
                if self.augmentations:
                    
                    image, mask = apply_transforms(image, mask, self.augmentations, self.add_noise, self.noise_params)
                
                batch_original_images.append(original_image)
                batch_images.append(image)
                batch_masks.append(mask)
                batch_shapes.append(original_shape)
                batch_paddings.append(padding)

                if len(batch_images) == self.batch_size:
                    yield batch_original_images, stack(batch_images), stack(batch_masks), batch_shapes, batch_paddings
                    batch_original_images = []
                    batch_images = []
                    batch_masks = []
                    batch_shapes = []
                    batch_paddings = []

        if batch_images:
            yield batch_original_images, stack(batch_images), stack(batch_masks), batch_shapes, batch_paddings

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

