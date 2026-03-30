#!/bin/bash
#SBATCH --job-name=aa_parallel_c100_4g
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

source ~/.bashrc
conda activate robustness

cd /home/tong.li003/adversarial_robustness_pytorch
export PYTHONPATH=/home/tong.li003/adversarial_robustness_pytorch:$PYTHONPATH

mkdir -p logs

python eval-aa-parallel.py \
  --data-dir ~/data/cifar100s \
  --log-dir ~/logs \
  --desc parallel-cifar100-wrn34-10-4g \
  --version standard

echo "AutoAttack evaluation completed."
