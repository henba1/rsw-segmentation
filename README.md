# Resistance Spot Welding Segmentation

A deep learning framework for semantic segmentation of resistance spot welding (RSW) nuggets in industrial imaging applications. This project implements and compares multiple state-of-the-art segmentation architectures for automated quality control in spot welding processes.

## Overview

This repository contains a complete machine learning pipeline for resistance spot welding segmentation, developed as part of research into automated quality control systems. The framework supports multiple deep learning architectures and provides tools for training, evaluation, and deployment in high-performance computing environments.

### Supported Models

The framework implements five segmentation architectures:

- **UNet**: Classic encoder-decoder architecture with skip connections
- **UNet++**: Enhanced UNet with nested dense skip pathways  
- **DeepLabV3+**: Atrous spatial pyramid pooling with encoder-decoder structure
- **SegFormer**: Vision transformer-based segmentation model
- **MiniUNet**: Lightweight variant optimized for resource-constrained environments

Each model supports configurable encoders (ResNet18, ResNet34, etc.) and can be trained with or without pretrained weights.

## Project Structure

```
rsw/
├── src/                          # Source code
│   ├── main.py                   # Main training script
│   ├── train.py                  # Training loop implementation
│   ├── test.py                   # Model evaluation
│   ├── PrepareData.py            # Data loading and preprocessing
│   ├── DataProcessing.py         # Data transformation pipeline
│   ├── BatchGenerator.py         # Batch generation with augmentation
│   ├── model_utils.py            # Model utilities and I/O
│   ├── metrics.py                # Evaluation metrics
│   ├── DatasetCreation.py        # Dataset preparation tools
│   ├── configs/                  # Model configurations
│   │   ├── *.json               # Standard training configs
│   │   └── prelim/              # Quick training configs
│   ├── scripts/                  # HPC job submission scripts
│   │   ├── submit_*.sh          # Model-specific submission scripts
│   └── run_job.sh               # Unified SLURM job script
├── data/                         # Data directory (excluded from git)
├── models/                       # Trained model checkpoints (excluded from git)
├── cluster_printouts/            # HPC job logs (excluded from git)
├── pyproject.toml               # Modern Python project configuration
├── requirements.txt             # Legacy requirements (HPC-specific)
└── README.md                    # This file
```

## Data Format and Processing

### Input Data Structure

The framework expects preprocessed data in numpy array format with the following structure:

```
data/
├── all_images_data_EUR.npy      # European dataset
└── all_images_data_lab.npy      # Laboratory dataset
```

Each data entry contains:
- `[0]` Original RSW image
- `[1]` Mask-overlayed image  
- `[2]` Image filename
- `[3]` Mask filename
- `[4]` Dataset path
- `[5]` Polygon annotation points
- `[6]` Class ID
- `[7]` Annotation state
- `[8]` Image dimensions (height, width)

### Data Preprocessing

The preprocessing pipeline includes:
- Image resizing to configurable dimensions (default: 512x512)
- Normalization and standardization
- Data augmentation (rotation, translation, noise injection)
- Train/validation/test splitting with stratification
- Cross-validation fold generation

## Configuration System

Model behavior is controlled through JSON configuration files in `src/configs/`. Each model has separate configurations for standard and preliminary (quick) training:

```json
{
    "model_type": "unet",
    "dataset": "EUR",
    "train_batch_size": 64,
    "num_epochs": 100,
    "lr": 1e-4,
    "model_enc": "resnet34",
    "resize_dim": 512,
    "augment_factor": 3
}
```

Key configuration parameters:
- **Model settings**: Architecture, encoder, pretrained weights
- **Training parameters**: Batch size, epochs, learning rate, optimizer settings
- **Data processing**: Augmentation, resize dimensions, noise parameters
- **Evaluation**: Binary threshold, test percentage, cross-validation folds

## Installation and Setup

### Standard Installation

```bash
# Clone the repository
git clone <repository-url>
cd rsw-segmentation

# Install with pip (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install additional HPC dependencies if needed
pip install -e ".[hpc]"
```

### Environment Configuration

Set the Comet ML API key for experiment tracking:

```bash
export COMET_API_KEY="your-comet-api-key"
```

## Usage

### Local Training

Basic model training:

```bash
cd src
python main.py <model_name> <preliminary> <use_pretrained>
```

Parameters:
- `model_name`: unet, unetplusplus, deeplabv3plus, segformer, miniunet
- `preliminary`: 0 (full training) or 1 (quick training)
- `use_pretrained`: 0 (from scratch) or 1 (pretrained encoder)

Examples:
```bash
# Train UNet with pretrained encoder
python main.py unet 0 1

# Quick training of SegFormer from scratch
python main.py segformer 1 0
```

### HPC Deployment

The framework includes optimized scripts for SLURM-based HPC clusters:

```bash
cd src

# Submit single job
sbatch run_job.sh unet 0 1

# Use convenience scripts
./scripts/submit_unet.sh 0 1
./scripts/submit_segformer.sh 1 1
```

The unified job script (`run_job.sh`) provides:
- Automatic parameter validation
- Resource allocation (GPU, memory, time limits)
- Module loading for HPC environments
- Comprehensive logging and error handling

### Monitoring and Evaluation

Training progress is tracked through:
- **Comet ML**: Experiment logging with metrics, hyperparameters, and artifacts
- **Local logs**: Console output and SLURM job files
- **Model checkpoints**: Automatic saving during training
- **Metrics files**: JSON-formatted evaluation results

Evaluation metrics include:
- Intersection over Union (IoU)
- Dice coefficient
- Pixel accuracy
- Binary classification metrics

## Results and Output

### Model Outputs

Trained models are saved in `models/` with the following structure:
- Model state dictionaries (`.pt` files)
- Training metrics (JSON format)
- Model visualizations (computational graphs)
- Experiment configurations

### Prediction Results

Test predictions are organized by model and timestamp:
```
preds_<ModelName>_<Timestamp>/
├── predictions/              # Segmentation masks
├── overlays/                # Prediction overlays
├── metrics.json             # Evaluation metrics
└── config.json              # Model configuration
```

## Technical Details

### Model Architectures

All models are implemented using:
- **segmentation-models-pytorch** for UNet variants and DeepLabV3+
- **transformers** library for SegFormer
- **timm** for backbone encoders
- **PyTorch** as the core framework

### Training Features

- **Mixed precision training** support for faster training
- **Learning rate scheduling** with ReduceLROnPlateau
- **Early stopping** based on validation metrics
- **Data augmentation** with configurable parameters
- **Cross-validation** support for robust evaluation

### Performance Optimization

- Efficient data loading with PyTorch DataLoader
- GPU memory optimization for large batch sizes
- Checkpoint saving for training interruption recovery
- Configurable batch sizes for different hardware

## Contributing

This research codebase follows standard software development practices:

- **Code formatting**: Black and isort for consistent style
- **Type checking**: MyPy for static analysis
- **Testing**: pytest framework for unit tests
- **Documentation**: Sphinx for API documentation

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rsw_segmentation,
  title={Resistance Spot Welding Segmentation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rsw-segmentation}
}
```

## Contact

For questions about this research or collaboration opportunities, please contact [your.email@example.com].
