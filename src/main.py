import json
import random
from comet_ml import Experiment
import torch
from torchvision import transforms
from PrepareData import PrepareData
from DataPreprocessing import DataPreprocessing
from BatchGenerator import BatchGenerator
from train import train_model
from test import test_model
from model_utils import load_model, save_model_visualization
from DeepLabV3Plus import DeepLabV3Plus
from UNet import UNet
from UNetPlusPlus import UNetPlusPlus
from SegFormer import SegFormer


def main():

    with open('../config.json', 'r') as f:
        config = json.load(f)

    experiment = Experiment(
        api_key="GeoZOdwTSNAEqugIyovCVq2Kv",
        project_name=config["project_name"],
        workspace=config["workspace"],
    )
 
    # 1 Prepare data
    prepare_data = PrepareData(dataset=config["dataset"], n_splits=config["n_splits"], random_state=config["random_state"], local=config["local"])
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
    
    experiment.log_parameter("dataset", config["dataset"])
    experiment.log_parameter("n_splits", config["n_splits"])
    experiment.log_parameter("random_state", config["random_state"])

    experiment.log_asset('../train_val_data.csv')
    experiment.log_asset('../test_data.csv')

    # Shuffle the data to avoid overfitting to the order of the data (we have two distinct datasets)
    combined_data = list(zip(trainval, trainval_names, trainval_labelmasks, trainval_idxs))
    random.shuffle(combined_data)
    trainval, trainval_names, trainval_labelmasks, trainval_idxs = zip(*combined_data)

    # 2 Preprocess the data
    custom_resize_transform = transforms.Resize((512, 512))
    train_dataset = DataPreprocessing(trainval, trainval_labelmasks, resize_transform=custom_resize_transform)
    test_dataset = DataPreprocessing(test, test_labelmasks, resize_transform=custom_resize_transform)

    data_augmentation_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=config["rotation_deg"], translate=(config["translation"], config["translation"]))
    ])

    # 3 Create batches 
    train_batch_generator = BatchGenerator(train_dataset, config["train_batch_size"], augmentations=data_augmentation_transforms, augment_factor=config["augment_factor"])
    test_batch_generator = BatchGenerator(test_dataset, config["test_batch_size"], augmentations=None, augment_factor=1)
    
    # 4 Initialize model
    if config["model_type"] == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=config["encoder_weights"] == "imagenet")
    elif config["model_type"] == "unet":
        model = UNet(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=config["encoder_weights"] == "imagenet")
    elif config["model_type"] == "unetplusplus":
        model = UNetPlusPlus(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=config["encoder_weights"] == "imagenet")
    elif config["model_type"] == "segformer":
        model = SegFormer(num_labels=1)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Save model visualization
    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    save_model_visualization(model, config["model_type"], dummy_input)

    
    # Log model parameters
    experiment.log_parameters({
        "model": config["model_type"],
        "in_channels": 1,
        "out_channels": 1,
        "learning_rate": config["lr"],
        "train_batch_size": config["train_batch_size"],
        "test_batch_size": config["test_batch_size"],
        "augment_factor": config["augment_factor"],
        "num_epochs": config["num_epochs"]
    })

    if config["loadModel"]:
        model, epoch = load_model(model)
        print(f"Model loaded from epoch {epoch}")
    else:
        # 5 Training 
        train_model(model, device, train_batch_generator, num_epochs=config["num_epochs"], lr=config["lr"], checkpoint_batch=20, experiment=experiment)
    
    # 6 Test model
    model_type = type(model).__name__
    test_model(model, device, test_batch_generator, test_names=test_names, test_idxs=test_idxs, model_name=model_type, experiment=experiment)


    # train_val splits for later (see code on cluster) 

if __name__ == "__main__":
    main()
