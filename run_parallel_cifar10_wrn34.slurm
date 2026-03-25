#!/bin/bash
#SBATCH --job-name=parallel_c10_wrn34
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Parallel Concat Fusion: CIFAR-10, WRN-34-10-Swish x 2
#
# Stage 1: CE pre-train 2 sub-models (vehicle/animal), 100 epochs
# Stage 2a: CE warmup fusion, 10 epochs
# Stage 2b: TRADES + EMA + CutMix, 400 epochs
#
# All Stage 2 hyperparameters match baseline (Rebuffi 2021)
# Parallel-specific: backbone_lr_ratio, aux_ce_loss
# ============================================================

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python train-wa-parallel.py \
  --data-dir ~/data/cifar10s \
  --log-dir ~/logs \
  --desc parallel-cifar10-wrn34-10-cutmix \
  --data cifar10s \
  --depth 34 --width 10 \
  --num-groups 2 \
  --batch-size 256 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --cutmix \
  --num-adv-epochs 400 \
  --epochs-sub 50 \
  --epochs-warmup 10 \
  --aux-weight 0.02 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
