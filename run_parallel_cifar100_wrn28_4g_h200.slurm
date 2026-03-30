#!/bin/bash
#SBATCH --job-name=parallel_c100_4g
#SBATCH --partition=AICORE_H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2g.35gb:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Parallel Concat Fusion: CIFAR-100, WRN-28-10-Swish x 4
#   Split: textured_organic / smooth_organic / rigid_manmade / large_structures
#
# Stage 1: CE pre-train 4 sub-models, 80 epochs
# Stage 2a: CE warmup fusion, 20 epochs
# Stage 2b: TRADES + EMA + MultiStepLR [200,300], 400 epochs
#
# H200 MIG version: 8x 2g.35gb instances, batch_size=128
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
  --desc parallel-cifar100-wrn28-10-4g \
  --data cifar100s \
  --depth 28 --width 10 \
  --num-groups 4 \
  --batch-size 128 \
  --lr 0.05 \
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
