#!/bin/bash
#SBATCH --job-name=cifar10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=./logs/cifar10.txt 
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl

cd ~/GitWS/JAX-vs-PyTorch-on-GPU/
python pytorch/run.py -b CIFAR10