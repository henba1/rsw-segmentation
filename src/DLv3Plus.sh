#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ssqc_DLv3Plus
#SBATCH --output=../cluster_printouts/prints_DLv3Plus.o%J
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


### Load modules
module load GCC
module load Python
module load CUDA

python main.py deeplabv3plus 0