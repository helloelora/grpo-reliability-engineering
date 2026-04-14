#!/bin/bash
#SBATCH --job-name=dpo-train
#SBATCH --output=logs/dpo_%j.log
#SBATCH --error=logs/dpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# DPO training - 233 preference pairs from base model generations

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_dpo"
mkdir -p "$WORKDIR/torch_cache"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_dpo_pairs.json"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_dpo":/tmp \
  --env TORCH_COMPILE_DISABLE=1 \
  --env TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache" \
  --env HF_HOME="$HF_HOME" \
  --env DATASET_PATH="$DATASET_PATH" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/dpo_train.py"

echo "Job finished at $(date)"
