#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ssqc_unet
#SBATCH --output=../cluster_printouts/prints_unet.o%J
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


### Load modules
module load GCC
module load Python
module load CUDA

python main.py unet 0 0