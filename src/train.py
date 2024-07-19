from comet_ml import Experiment
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from metrics import Metrics
from model_utils import save_model, verify_predictions_and_labels, save_metrics_to_json
import json


def train_model(model, device, train_loader, val_loader=None, num_epochs=50, lr=1e-4, checkpoint_batch=50, model_name='UNet', experiment=None, verify=False, up_to_epoch=0, prelim=False): 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    criterion = nn.BCELoss()
    print("Model was trained up to epoch:")
    print(up_to_epoch)
    print("==============================")
    print(f"Using device: {device}")

    total_batches = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Training loop
        train_iou_scores = []
        train_dice_scores = []
        train_accuracies = []
        train_losses = []
        
        
        for batch_idx, (_, images, masks, _, _) in enumerate(train_loader):
            total_batches += 1
            batch_start_time = time.time()

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            if verify:
                verify_predictions_and_labels(preds, masks)

            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_iou_scores.append(Metrics.mean_iou(preds, masks))
            train_dice_scores.append(Metrics.dice_coefficient(preds, masks))
            train_accuracies.append(Metrics.accuracy(preds, masks))

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            # print(f'Batch {batch_idx + 1} processed in {batch_duration:.4f} seconds')
            # print(f'Training loss: {loss.item()}')
            # print(train_dice_scores[-1])
            # print(train_iou_scores[-1])

            if experiment:
                experiment.log_metric("batch_train_loss", loss.item(), step=epoch+up_to_epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_iou", train_iou_scores[-1], step=epoch+up_to_epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_dice", train_dice_scores[-1], step=epoch+up_to_epoch * len(train_loader) + batch_idx)
                experiment.log_metric("batch_train_accuracy", train_accuracies[-1], step=epoch+up_to_epoch * len(train_loader) + batch_idx)

            # Save model after every checkpoint_batch amount of batches
            if total_batches > 0 and total_batches % checkpoint_batch == 0:
                save_model(model, epoch+up_to_epoch, directory="../models", prelim=prelim)
        
        epoch_train_loss = np.mean(train_losses)
        epoch_train_iou = np.mean(train_iou_scores)
        epoch_train_dice = np.mean(train_dice_scores)
        epoch_train_accuracy = np.mean(train_accuracies)

        print(f'Epoch {epoch + 1 + up_to_epoch} Training Loss: {epoch_train_loss:.4f}')
        print(f'Epoch {epoch + 1 + up_to_epoch} Training IoU: {epoch_train_iou:.4f}')
        print(f'Epoch {epoch + 1+ up_to_epoch} Training Dice: {epoch_train_dice:.4f}')
        print(f'Epoch {epoch + 1+ up_to_epoch} Training Accuracy: {epoch_train_accuracy:.4f}')

        if experiment:
            experiment.log_metric("epoch_train_iou", epoch_train_iou, step=epoch+up_to_epoch)
            experiment.log_metric("epoch_train_dice", epoch_train_dice, step=epoch+up_to_epoch)
            experiment.log_metric("epoch_train_accuracy", epoch_train_accuracy, step=epoch+up_to_epoch)
        
        # Step the scheduler based on the training loss
        scheduler.step(epoch_train_loss)
        
        # Save model after every epoch
        save_model(model, epoch+up_to_epoch, directory="../models", prelim=prelim)

        # Validation loop
        if val_loader:
            model.eval()
            val_iou_scores = []
            val_dice_scores = []
            val_accuracies = []
            val_losses = []

            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        images, masks, _ = batch_data
                    else:
                        images, masks = batch_data

                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)

                    val_loss = criterion(preds, masks)
                    val_losses.append(val_loss.item())

                    val_iou_scores.append(Metrics.mean_iou(preds, masks))
                    val_dice_scores.append(Metrics.dice_coefficient(preds, masks))
                    val_accuracies.append(Metrics.accuracy(preds, masks))

            epoch_val_iou = np.mean(val_iou_scores)
            epoch_val_dice = np.mean(val_dice_scores)
            epoch_val_accuracy = np.mean(val_accuracies)
            epoch_val_loss = np.mean(val_losses)

            print(f'Epoch {epoch + 1 + up_to_epoch} Validation IoU: {epoch_val_iou:.4f}')
            print(f'Epoch {epoch + 1 + up_to_epoch} Validation Dice: {epoch_val_dice:.4f}')
            print(f'Epoch {epoch + 1 + up_to_epoch} Validation Accuracy: {epoch_val_accuracy:.4f}')

            if experiment:
                experiment.log_metric("epoch_val_iou", epoch_val_iou, step=epoch+up_to_epoch)
                experiment.log_metric("epoch_val_dice", epoch_val_dice, step=epoch+up_to_epoch)
                experiment.log_metric("epoch_val_accuracy", epoch_val_accuracy, step=epoch+up_to_epoch)

            # Step the scheduler based on the validation loss
            scheduler.step(epoch_val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch + 1 + up_to_epoch} completed in {epoch_duration:.4f} seconds')
        if num_epochs==1:
            with open(f'../configs/{model_name}_config.json', 'r') as f:
                config = json.load(f)
            desired_epochs = config['num_epochs']
            est_time = desired_epochs*epoch_duration/ 60
            print(f'Estimated training time of the model: {est_time:.4f} minutes')
            
        if experiment:
            experiment.log_metric("epoch_duration", epoch_duration, step=epoch)
        metrics = {
            "epoch": epoch + 1 + up_to_epoch,
            "train_iou": epoch_train_iou,
            "train_dice": epoch_train_dice,
            "train_accuracy": epoch_train_accuracy,
            "val_iou": epoch_val_iou if val_loader else None,
            "val_dice": epoch_val_dice if val_loader else None,
            "val_accuracy": epoch_val_accuracy if val_loader else None,
        }
        
        
        save_metrics_to_json(metrics, type(model).__name__, directory='../models', prelim=prelim)

def check_GPU():
    if torch.cuda.is_available():
        # Perform a dummy operation to allocate some memory on the GPU
        x = torch.rand(10000, 10000).cuda()
        
        print('========================================================================')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print('========================================================================')
        
        # Clean up afterwards
        del x
        torch.cuda.empty_cache()
    else:
        print('========================================================================')
        print("No GPU available, using CPU instead.")
        print('========================================================================')
