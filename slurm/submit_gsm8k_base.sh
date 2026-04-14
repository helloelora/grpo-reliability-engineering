#!/bin/bash
#SBATCH --job-name=gsm8k-base
#SBATCH --output=logs/gsm8k_base_%j.log
#SBATCH --error=logs/gsm8k_base_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# GSM8K benchmark - Qwen3-8B base (no LoRA)
# 1319 questions, ~6-10h on A100

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_gsm8k_base"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH=""
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export RUN_NAME="qwen3_base"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_gsm8k_base":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_gsm8k.py"

echo "Job finished at $(date)"
