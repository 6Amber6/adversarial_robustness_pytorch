#!/bin/bash
#SBATCH --job-name=rebuffi_wrn34_cutmix
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

source ~/.bashrc
conda activate base

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python train-wa.py \
  --data-dir ~/data/cifar10s \
  --log-dir ~/logs \
  --desc wrn34-10-cutmix-beta11 \
  --data cifar10s \
  --batch-size 1024 \
  --model wrn-34-10-swish \
  --num-adv-epochs 400 \
  --lr 0.4 \
  --beta 6.0 \
  --unsup-fraction 0.0 \
  --aux-data-filename None \
  --cutmix
