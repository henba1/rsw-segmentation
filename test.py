from comet_ml import Experiment
import numpy as np
import torch
from metrics import Metrics
from DataPreprocessing import DataPreprocessing, visual_inspect
import os
import datetime

def test_model(model, device, test_loader, test_names, test_idxs, model_name, experiment=None):
    # Set model to evaluation mode
    model.eval()
    iou_scores = []
    dice_scores = []
    accuracies = []

    # Create directory for saving predictions
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"../preds_{model_name}_{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks, original_shapes) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for i in range(len(preds)):
                pred = preds[i]
                mask = masks[i]
                original_shape = original_shapes[i]

                # Ensure original_shape is a tuple of integers
                if isinstance(original_shape, torch.Tensor):
                    original_shape = tuple(original_shape.cpu().numpy())
                else:
                    original_shape = tuple(original_shape)

                pred_resized = DataPreprocessing.unpad_and_resize(pred, original_shape)
                mask_resized = DataPreprocessing.unpad_and_resize(mask, original_shape)

                # Compute metrics
                iou_scores.append(Metrics.mean_iou(pred_resized, mask_resized))
                dice_scores.append(Metrics.dice_coefficient(pred_resized, mask_resized))
                accuracies.append(Metrics.accuracy(pred_resized, mask_resized))

                # Get the original image name and index
                image_name = test_names[idx * len(preds) + i]
                image_idx = test_idxs[idx * len(preds) + i]

                # Save overlay images
                original_image = images[i].cpu()
                visual_inspect(original_image, pred_resized, save_path=os.path.join(save_dir, f"pred_{image_idx}_{image_name}"))
                visual_inspect(original_image, mask_resized, save_path=os.path.join(save_dir, f"gt_{image_idx}_{image_name}"))

    mean_iou_score = np.mean(iou_scores)
    mean_dice_score = np.mean(dice_scores)
    mean_accuracy = np.mean(accuracies)

    print(f"Mean IoU: {mean_iou_score:.4f}")
    print(f"Dice Coefficient: {mean_dice_score:.4f}")
    print(f"Accuracy: {mean_accuracy:.4f}")

    if experiment:
        experiment.log_metric("mean_iou", mean_iou_score)
        experiment.log_metric("dice_coefficient", mean_dice_score)
        experiment.log_metric("accuracy", mean_accuracy)
        experiment.log_asset_folder(save_dir)
