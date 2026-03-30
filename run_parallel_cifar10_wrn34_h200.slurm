#!/bin/bash
#SBATCH --job-name=parallel_c10_wrn34
#SBATCH --partition=AICORE_H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2g.35gb:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Parallel Concat Fusion: CIFAR-10, WRN-34-10-Swish x 2
#
# Stage 1: CE pre-train 2 sub-models (vehicle/animal), 80 epochs
# Stage 2a: CE warmup fusion, 10 epochs
# Stage 2b: TRADES + EMA + MultiStepLR [200,300], 400 epochs
#
# H200 MIG version: 6x 2g.35gb instances, batch_size=128
# ============================================================

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

# MIG instances do not support NCCL P2P
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

mkdir -p logs

python train-wa-parallel.py \
  --data-dir ~/data/cifar10s \
  --log-dir ~/logs \
  --desc parallel-cifar10-wrn34-10 \
  --data cifar10s \
  --depth 34 --width 10 \
  --num-groups 2 \
  --batch-size 128 \
  --lr 0.05 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --num-adv-epochs 400 \
  --scheduler step --scheduler-milestones 200 300 \
  --epochs-sub 80 \
  --epochs-warmup 10 \
  --aux-weight 0.02 \
  --clip-grad 5.0 \
  --bn-freeze-epochs 100 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
