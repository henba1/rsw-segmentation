from comet_ml import Experiment
import numpy as np
import torch
from torchvision import transforms
from DataPreprocessing import DataPreprocessing
from BatchGenerator import BatchGenerator
from PrepareData import PrepareData
import random
from UNet import UNet
import segmentation_models_pytorch as smp
from train import train_model
from test import test_model



def main():
    #====================
    train_batch_size = 8
    test_batch_size = 8
    rotation_deg = 20
    translation = 0.1
    lr = 1e-4
    #====================

    experiment = Experiment(
        api_key="GeoZOdwTSNAEqugIyovCVq2Kv",
        project_name="ssqc-rsw-1-local",
        workspace="simon-onyx",
    )

    # Prepare data
    prepare_data = PrepareData(dataset='EUR', n_splits=6, random_state=50)
    trainval, trainval_names, trainval_labelmasks, trainval_idxs, test, test_names, test_labelmasks, test_idxs, data_list, dfs_img_has_mask, df_trainval, df_test, folds = (
        prepare_data.trainval,
        prepare_data.trainval_names,
        prepare_data.trainval_labelmasks,
        prepare_data.trainval_idxs,
        prepare_data.test,
        prepare_data.test_names,
        prepare_data.test_labelmasks,
        prepare_data.test_idxs,
        prepare_data.data_list,
        prepare_data.dfs_img_has_mask,
        prepare_data.df_train_val,
        prepare_data.df_test,
        prepare_data.folds
    )
    
    experiment.log_parameter("dataset", "both")
    experiment.log_parameter("n_splits", 6)
    experiment.log_parameter("random_state", 50)

    experiment.log_asset('../train_val_data.csv')
    experiment.log_asset('../test_data.csv')

    combined_data = list(zip(trainval, trainval_names, trainval_labelmasks, trainval_idxs))
    random.shuffle(combined_data)
    trainval, trainval_names, trainval_labelmasks, trainval_idxs = zip(*combined_data)

    # Preprocess the data
    custom_resize_transform = transforms.Resize((512, 512))
    train_dataset = DataPreprocessing(trainval, trainval_labelmasks, resize_transform=custom_resize_transform)
    test_dataset = DataPreprocessing(test, test_labelmasks, resize_transform=custom_resize_transform)

    data_augmentation_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=rotation_deg, translate=(translation,translation))
    ])

    # Create batches 
    train_batch_generator = BatchGenerator(train_dataset, train_batch_size, augmentations=data_augmentation_transforms, augment_factor=2)
    test_batch_generator = BatchGenerator(test_dataset, test_batch_size, augmentations=None, augment_factor=1)
    
    #====================
    model = smp.Unet(
    encoder_name="resnet34",        # Choose an encoder from the segmentation_models_pytorch library
    encoder_weights="imagenet",     # Use pretrained weights from ImageNet
    in_channels=1,                  # Model input channels (1 for grayscale images)
    classes=1                       # Model output channels (1 for binary segmentation)
)
    #model = UNet(in_channels=1, out_channels=1)
    #====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Log model parameters
    experiment.log_parameters({
        "model": "UNet",
        "in_channels": 1,
        "out_channels": 1,
        "learning_rate": lr,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "augment_factor": 2,
        "num_epochs": 1
    })
    
    # Training loop
    train_model(model, device, train_batch_generator, num_epochs=1, lr=lr, experiment=experiment)

    # Test the model
    #test_model(model, device, test_batch_generator, experiment=experiment)

    
    #train_val split implementation for later

if __name__ == "__main__":
    main()
