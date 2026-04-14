#!/bin/bash
#SBATCH --job-name=sft-qwen25
#SBATCH --output=logs/sft_qwen25_%j.log
#SBATCH --error=logs/sft_qwen25_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# SFT Qwen2.5-7B — 5-fold CV, 190 questions, 3 epochs per fold

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_sft_qwen25"
mkdir -p "$WORKDIR/torch_cache"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_sft_combined.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_sft_qwen25":/tmp \
  --env TORCH_COMPILE_DISABLE=1 \
  --env TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_cache" \
  --env HF_HOME="$HF_HOME" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/sft_train_qwen25.py"

echo "Job finished at $(date)"
