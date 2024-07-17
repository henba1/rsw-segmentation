#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ssqc_rsw
#SBATCH --output=./prints_rsw.o%J
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G


### Load modules
module load Python/3.9.6
module load cuDNN/8.6.0.163-CUDA-11.8.0    
module load GCCcore/.9.3.0

python ./src/main.py