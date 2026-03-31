#!/bin/bash
#SBATCH --job-name=grpo-train
#SBATCH --output=logs/grpo_%j.log
#SBATCH --error=logs/grpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00

# ============================================================
# GRPO V3 Training — Qwen3-8B on 57 hard questions
# 21 from master_dataset + 36 from hard_numeric_generated
# Improved regex reward (%, \times, fractions, commas)
# ============================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_grpo"
mkdir -p "$WORKDIR/torch_cache"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_combined.json"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_grpo":/tmp \
  --env TORCH_COMPILE_DISABLE=1 \
  --env TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache" \
  --env HF_HOME="$HF_HOME" \
  --env DATASET_PATH="$DATASET_PATH" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/grpo_train.py"

echo "=== Job finished at $(date) ==="
