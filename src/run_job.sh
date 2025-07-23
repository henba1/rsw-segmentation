#!/usr/local_rwth/bin/zsh

# Unified HPC Job Script for RSW Segmentation Models
# Usage: sbatch run_job.sh <model_name> <preliminary_training> <use_pretrained>
# Example: sbatch run_job.sh unet 0 1

#SBATCH --job-name=rsw-segmentation
#SBATCH --output=../cluster_printouts/prints_%x_%j.out
#SBATCH --error=../cluster_printouts/prints_%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Check if required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <preliminary_training> <use_pretrained>"
    echo "model_name: unet, unetplusplus, deeplabv3plus, segformer, miniunet"
    echo "preliminary_training: 0 (normal) or 1 (quick)"
    echo "use_pretrained: 0 (from scratch) or 1 (pretrained)"
    exit 1
fi

MODEL_NAME=$1
PRELIMINARY=$2
USE_PRETRAINED=$3

# Validate model name
case $MODEL_NAME in
    unet|unetplusplus|deeplabv3plus|segformer|miniunet)
        echo "Training $MODEL_NAME model..."
        ;;
    *)
        echo "Error: Invalid model name '$MODEL_NAME'"
        echo "Valid options: unet, unetplusplus, deeplabv3plus, segformer, miniunet"
        exit 1
        ;;
esac

# Load required modules
module load GCC
module load Python
module load CUDA

# Create output directory if it doesn't exist
mkdir -p ../cluster_printouts

# Log job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
echo "Preliminary training: $PRELIMINARY"
echo "Use pretrained: $USE_PRETRAINED"
echo "================================"

# Run the training script
python main.py $MODEL_NAME $PRELIMINARY $USE_PRETRAINED

echo "================================"
echo "Job completed at: $(date)" 