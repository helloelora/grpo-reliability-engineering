#!/bin/bash
#SBATCH --job-name=grpo-qwen14b
#SBATCH --output=logs/grpo_%j.log
#SBATCH --error=logs/grpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# ============================================================
# GRPO Training - Qwen3-14B Reliability Engineering
# LoRA r=32, G=4 completions, β=0.001, 200 steps per fold
# 5-fold CV with Claude 3.5 Sonnet judge
# ============================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

# Create temp directories
mkdir -p $WORKDIR/mytmp_grpo
mkdir -p $WORKDIR/fine_tuning_qwen/logs

export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface
export DATASET_PATH=$WORKDIR/fine_tuning_qwen/data/dataset_alex.json

# GRPO generates multiple completions → needs more time
# 5 folds × 200 steps × 4 generations = heavy compute
apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_grpo:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/training/grpo_train.py

echo "=== Job finished at $(date) ==="
