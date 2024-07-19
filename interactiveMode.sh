#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=ssqc_rsw
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10

# Load necessary modules
module load GCC
module load Python
module load CUDA
