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
# Parallel Concat Fusion: CIFAR-100, WRN-34-10-Swish x 4
#   Split: textured_organic / smooth_organic / rigid_manmade / large_structures
#   (25 classes each, semantic superclass grouping)
#
# Stage 1: CE pre-train 4 sub-models, 80 epochs
# Stage 2a: CE warmup fusion, 20 epochs
# Stage 2b: TRADES + EMA + MultiStepLR [200,300], 400 epochs
#
# NOTE: 4 sub-models (~184M params), uses 3 GPUs
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
  --desc parallel-cifar100-wrn34-10-4g \
  --data cifar100s \
  --depth 34 --width 10 \
  --num-groups 4 \
  --batch-size 256 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --num-adv-epochs 400 \
  --scheduler step --scheduler-milestones 200 300 \
  --epochs-sub 80 \
  --epochs-warmup 20 \
  --aux-weight 0.1 \
  --clip-grad 5.0 \
  --bn-freeze-epochs -1 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
