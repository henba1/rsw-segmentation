import numpy as np
import torch
from comet_ml import Experiment
from metrics import Metrics
from DataProcessing import DataProcessing, visual_inspect
import os
import datetime

def test_model(model, device, test_loader, test_names, test_idxs, test_dims, test_materials, resize_dim, model_name, bin_thresh=0.5, experiment=None):
    model.eval()
    iou_scores = []
    dice_scores = []
    accuracies = []

    # Create directory for saving predictions
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"../preds_{model_name}_{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (original_images, images, masks, original_shapes, paddings) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for i in range(len(preds)):
                original_image = original_images[i]
                image = images[i]
                pred = preds[i]
                mask = masks[i]
                original_shape = original_shapes[i]
                padding = paddings[i]

                image_resized = DataProcessing.unpad_and_resize(image, original_shape, padding)
                pred_resized = DataProcessing.unpad_and_resize(pred, original_shape, padding)
                mask_resized = DataProcessing.unpad_and_resize(mask, original_shape, padding)

                # Compute metrics
                iou_scores.append(Metrics.mean_iou(pred_resized, mask_resized))
                dice_scores.append(Metrics.dice_coefficient(pred_resized, mask_resized))
                accuracies.append(Metrics.accuracy(pred_resized, mask_resized))

                # Get the original image name and index
                image_name = test_names[idx * len(preds) + i]
                image_idx = test_idxs[idx * len(preds) + i]
                image_material = test_materials[idx * len(preds) + i]

                # Save overlay images
                visual_inspect(image_resized, mask_resized, pred_tensor=pred_resized, resize_dim=resize_dim, save_path=os.path.join(save_dir, image_material, f"{image_name}_{image_idx}.png"), original_shape=original_shape, bin_thresh=bin_thresh) 

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
