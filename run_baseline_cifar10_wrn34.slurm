#!/bin/bash
#SBATCH --job-name=gowal_c10_wrn34
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Gowal et al., 2020 Baseline: CIFAR-10, WRN-34-10-Swish
# TRADES + EMA, cosinew scheduler, no CutMix, no extra data
# ============================================================

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python train-wa.py \
  --data-dir ~/data/cifar10s \
  --log-dir ~/logs \
  --desc gowal-cifar10-wrn34-10 \
  --data cifar10s \
  --model wrn-34-10-swish \
  --batch-size 256 \
  --lr 0.1 \
  --beta 6.0 \
  --weight-decay 5e-4 \
  --tau 0.999 \
  --num-adv-epochs 400 \
  --unsup-fraction 0.0 \
  --aux-data-filename None

echo "Training completed."
