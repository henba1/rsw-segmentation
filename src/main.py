import json
import random
import argparse
from comet_ml import Experiment
import torch
from torchvision import transforms
from PrepareData import PrepareData
from DataProcessing import DataProcessing
from BatchGenerator import BatchGenerator
from train import train_model, check_GPU
from test import test_model
from model_utils import get_config, load_model, save_model_visualization
from DeepLabV3Plus import DeepLabV3Plus
from UNet import UNet
from MiniUNet import MiniUNet
from UNetPlusPlus import UNetPlusPlus
from SegFormer import SegFormer

def main():

    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., deeplabv3plus, unet, etc.)')
    parser.add_argument('preliminary_training', type=int, help='Shorter training time (different config parameters for each model) for quicker results : 0 for normal training 1 for quicker training')
    parser.add_argument('use_pretrained', type=int, default=1, help='Use pretrained weights: 1 for pretrained, 0 for training from scratch')

    args = parser.parse_args()

    prelim = args.preliminary_training == 1
    if prelim:
        print('PRELIMINARY TRAINING SELECTED')

    use_pretrained = args.use_pretrained == 1

    # Initialize model and get config 
    if args.model_name == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=1, out_channels=1, encoder_name="resnet18", use_pretrained=use_pretrained)
        config, model_type = get_config(model, prelim)
        model = DeepLabV3Plus(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=use_pretrained)
    elif args.model_name == "unet":
        model = UNet(in_channels=1, out_channels=1, encoder_name="resnet18", use_pretrained=use_pretrained)
        config, model_type  = get_config(model, prelim)
        model = UNet(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=use_pretrained)
    elif args.model_name == "unetplusplus":
        model = UNetPlusPlus(in_channels=1, out_channels=1, encoder_name="resnet18", use_pretrained=use_pretrained)
        config, model_type  = get_config(model, prelim)
        model = UNetPlusPlus(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=use_pretrained)
    elif args.model_name == "segformer":
        model = SegFormer(num_labels=1, use_pretrained=use_pretrained)
        config, model_type  = get_config(model, prelim)
    elif args.model_name == "miniunet":
        # if use_pretrained:
        #     print("MiniUNet is not available with a pre-trained encoder.")
        #     exit(1)
        model = MiniUNet(in_channels=1, out_channels=1, encoder_name="resnet18", use_pretrained=use_pretrained)
        config, model_type  = get_config(model, prelim)
        model = MiniUNet(in_channels=1, out_channels=1, encoder_name=config["model_enc"], use_pretrained=use_pretrained)

    npy_path = config.get("npy_path", None)

    experiment = Experiment(
        api_key="GeoZOdwTSNAEqugIyovCVq2Kv",
        project_name=f"{config['project_name']}{config['model_type']}",
        workspace=config["workspace"],
        log_env_details=True,
        log_env_gpu=True,
        log_env_cpu=True   
    )
 
    # 1 Prepare data
    prepare_data = PrepareData(dataset=config["dataset"], test_perc=config['test_perc'], n_splits=config["n_splits"], random_state=config["random_state"], local=config["local"], npy_path=npy_path)
    trainval, trainval_names, trainval_labelmasks, trainval_idxs, trainval_dims, trainval_materials, test, test_names, test_labelmasks, test_idxs, test_dims, test_materials, data_list, dfs_img_has_mask, df_trainval, df_test, folds = (
        prepare_data.trainval,
        prepare_data.trainval_names,
        prepare_data.trainval_labelmasks,
        prepare_data.trainval_idxs,
        prepare_data.trainval_dims,
        prepare_data.trainval_materials,
        prepare_data.test,
        prepare_data.test_names,
        prepare_data.test_labelmasks,
        prepare_data.test_idxs,
        prepare_data.test_dims,
        prepare_data.test_materials,
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
    if prelim:
        experiment.log_asset(f"configs/prelim/{model_type}_config.json")
    else:
        experiment.log_asset(f"configs/{model_type}_config.json")


    # Shuffle the data to avoid overfitting to the order of the data (we have two distinct datasets)
    combined_data = list(zip(trainval, trainval_names, trainval_labelmasks, trainval_idxs, trainval_dims, trainval_materials))
    random.shuffle(combined_data)
    trainval, trainval_names, trainval_labelmasks, trainval_idxs, trainval_dims, trainval_materials = zip(*combined_data)
    
    # also shuffle the test data (for visual inspection purposes), can remove that later
    # combined_data = list(zip(test, test_names, test_labelmasks, test_idxs, test_dims, test_materials))
    # random.shuffle(combined_data)
    # test, test_names, test_labelmasks, test_idxs, test_dims, test_materials = zip(*combined_data)

    # 2 Preprocess the data
    if config['resize_dim']:
        resize_dim = config['resize_dim']
    else:
        resize_dim = 512
    custom_resize_transform = transforms.Resize((resize_dim, resize_dim))

    train_dataset = DataProcessing(trainval, trainval_labelmasks, trainval_dims, resize_transform=custom_resize_transform)
    test_dataset = DataProcessing(test, test_labelmasks, test_dims, resize_transform=custom_resize_transform)

    data_augmentation_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=config["rotation_deg"], translate=(config["translation"], config["translation"]))
    ])

    # 3 Create batches 
    train_batch_generator = BatchGenerator(
                            train_dataset, 
                            config["train_batch_size"], 
                            augmentations=data_augmentation_transforms, 
                            augment_factor=config["augment_factor"], 
                            add_noise=config['add_noise'], 
                            noise_params=config["noise_params"])
    
    test_batch_generator = BatchGenerator(test_dataset, config["test_batch_size"], augmentations=None, augment_factor=1)
    
    
    check_GPU()
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

    model_type = type(model).__name__

    if config["loadModel"]:
        model, up_to_epoch = load_model(model, prelim=prelim)
        print(f"Model loaded from epoch {up_to_epoch}")
    else:
        if config['estimate_train_time']:
            train_model(model, device, train_batch_generator, num_epochs=1, lr=config["lr"], checkpoint_batch=config['checkpoint_batch'], model_name=model_type, experiment=experiment, verify=False, up_to_epoch=0, prelim=prelim)
        else:
            # 4 Training 
            train_model(model, device, train_batch_generator, num_epochs=config["num_epochs"], lr=config["lr"], checkpoint_batch=config['checkpoint_batch'], model_name=model_type, experiment=experiment, verify=False, up_to_epoch=0, prelim=prelim)
    
    if config['continue_training']:
        # 4 Training 
        train_model(model, device, train_batch_generator, num_epochs=config["num_epochs"], lr=config["lr"], checkpoint_batch=config['checkpoint_batch'], model_name=model_type, experiment=experiment, verify=False, up_to_epoch=up_to_epoch, prelim=prelim)

    # 5 Test model
    if config['testModel']:
        test_model(
            model, 
            device, 
            test_batch_generator, 
            test_names=test_names, 
            test_idxs=test_idxs, 
            test_dims=test_dims, 
            test_materials=test_materials, 
            resize_dim=resize_dim, 
            model_name=model_type, 
            bin_thresh=config['bin_thresh'], 
            experiment=experiment, 
            groupby_material=config['groupby_material']
        )

    # # train_val splits for later (folds are implemented: prepare_data.folds)

if __name__ == "__main__":
    main()
