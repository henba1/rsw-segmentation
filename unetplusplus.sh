#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ssqc_unetPP
#SBATCH --output=./prints_unetPP.o%J
#SBATCH --time=9:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


### Load modules
module load GCC
module load Python
module load CUDA

python ./src/main.py unetplusplus