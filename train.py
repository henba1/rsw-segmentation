from comet_ml import Experiment
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
from metrics import Metrics
from model_utils import save_model

def train_model(model, device, train_loader, val_loader=None, num_epochs=1, lr=1e-4, checkpoint_batch=50, experiment=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Using device: {device}")

    total_batches = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Training loop
        train_iou_scores = []
        train_dice_scores = []
        train_accuracies = []

        for batch_idx, batch_data in enumerate(train_loader):
            total_batches += 1
            batch_start_time = time.time()

            if len(batch_data) == 3:
                images, masks, _ = batch_data
            else:
                images, masks = batch_data
            
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_iou_scores.append(Metrics.mean_iou(preds, masks))
            train_dice_scores.append(Metrics.dice_coefficient(preds, masks))
            train_accuracies.append(Metrics.accuracy(preds, masks))

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            print(f'Batch {batch_idx + 1} processed in {batch_duration:.4f} seconds')
            print(f'Training loss: {loss.item()}')

            if experiment:
                experiment.log_metric("batch_train_loss", loss.item(), step=epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_iou", train_iou_scores[-1], step=epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_dice", train_dice_scores[-1], step=epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_accuracy", train_accuracies[-1], step=epoch * len(train_loader) + batch_idx)

            # Save model after every checkpoint_batch amount of batches
            if total_batches > 0 and total_batches % checkpoint_batch == 0:
                save_model(model, epoch, directory="../models")

        epoch_train_iou = np.mean(train_iou_scores)
        epoch_train_dice = np.mean(train_dice_scores)
        epoch_train_accuracy = np.mean(train_accuracies)

        print(f'Epoch {epoch + 1} Training IoU: {epoch_train_iou:.4f}')
        print(f'Epoch {epoch + 1} Training Dice: {epoch_train_dice:.4f}')
        print(f'Epoch {epoch + 1} Training Accuracy: {epoch_train_accuracy:.4f}')

        if experiment:
            experiment.log_metric("epoch_train_iou", epoch_train_iou, step=epoch)
            experiment.log_metric("epoch_train_dice", epoch_train_dice, step=epoch)
            experiment.log_metric("epoch_train_accuracy", epoch_train_accuracy, step=epoch)

        # Save model after every epoch
        save_model(model, epoch, directory="../models")

        # Validation loop
        if val_loader:
            model.eval()
            val_iou_scores = []
            val_dice_scores = []
            val_accuracies = []

            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                    else:
                        images, masks = batch_data

                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)

                    val_iou_scores.append(Metrics.mean_iou(preds, masks))
                    val_dice_scores.append(Metrics.dice_coefficient(preds, masks))
                    val_accuracies.append(Metrics.accuracy(preds, masks))

            epoch_val_iou = np.mean(val_iou_scores)
            epoch_val_dice = np.mean(val_dice_scores)
            epoch_val_accuracy = np.mean(val_accuracies)

            print(f'Epoch {epoch + 1} Validation IoU: {epoch_val_iou:.4f}')
            print(f'Epoch {epoch + 1} Validation Dice: {epoch_val_dice:.4f}')
            print(f'Epoch {epoch + 1} Validation Accuracy: {epoch_val_accuracy:.4f}')

            if experiment:
                experiment.log_metric("epoch_val_iou", epoch_val_iou, step=epoch)
                experiment.log_metric("epoch_val_dice", epoch_val_dice, step=epoch)
                experiment.log_metric("epoch_val_accuracy", epoch_val_accuracy, step=epoch)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.4f} seconds')
        if experiment:
            experiment.log_metric("epoch_duration", epoch_duration, step=epoch)
