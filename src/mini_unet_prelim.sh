#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ssqc_Prelim_mini_unet
#SBATCH --output=../cluster_printouts/prints_Prelim_mini_unet.o%J
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


### Load modules
module load GCC
module load Python
module load CUDA

python main.py miniunet 1 0