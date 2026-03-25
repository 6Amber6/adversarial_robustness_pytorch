#!/bin/bash
#SBATCH --job-name=parallel_c100_wrn28_4g
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Parallel Concat Fusion: CIFAR-100, WRN-28-10-Swish x 4
#   Split: textured_organic / smooth_organic / rigid_manmade / large_structures
#   (25 classes each, semantic superclass grouping)
#
# Stage 1: CE pre-train 4 sub-models, 100 epochs
# Stage 2a: CE warmup fusion, 10 epochs
# Stage 2b: TRADES + EMA + CutMix, 400 epochs
#
# NOTE: 4 sub-models (~146M params), uses 3 GPUs
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
  --desc parallel-cifar100-wrn28-10-4g-cutmix \
  --data cifar100s \
  --depth 28 --width 10 \
  --num-groups 4 \
  --batch-size 256 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --cutmix \
  --num-adv-epochs 400 \
  --epochs-sub 80 \
  --epochs-warmup 20 \
  --aux-weight 0.0 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
