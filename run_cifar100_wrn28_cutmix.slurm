#!/bin/bash
#SBATCH --job-name=rebuffi_wrn28_cutmix_cifar100
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python train-wa.py \
  --data-dir ~/data/cifar100s \
  --log-dir ~/logs \
  --desc wrn28-10-cutmix-cifar100 \
  --data cifar100s \
  --batch-size 256 \
  --model wrn-28-10-swish \
  --num-adv-epochs 400 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --unsup-fraction 0.0 \
  --aux-data-filename None \
  --cutmix