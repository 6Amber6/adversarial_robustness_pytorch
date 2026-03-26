#!/bin/bash
#SBATCH --job-name=parallel_c100_wrn28_2g
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Parallel Concat Fusion: CIFAR-100, WRN-28-10-Swish x 2
#   Split: nature (50 classes) / manmade (50 classes)
#
# Stage 1: CE pre-train 2 sub-models, 100 epochs
# Stage 2a: CE warmup fusion, 10 epochs
# Stage 2b: TRADES + EMA + CutMix, 400 epochs
# ============================================================

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python train-wa-parallel.py \
  --data-dir ~/data/cifar100s \
  --log-dir ~/logs \
  --desc parallel-cifar100-wrn28-10-2g-cutmix \
  --data cifar100s \
  --depth 28 --width 10 \
  --num-groups 2 \
  --batch-size 256 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --cutmix \
  --num-adv-epochs 400 \
  --epochs-sub 100 \
  --epochs-warmup 10 \
  --aux-weight 0.02 \
  --clip-grad 5.0 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
