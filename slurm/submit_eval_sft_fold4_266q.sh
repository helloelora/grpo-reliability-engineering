#!/bin/bash
#SBATCH --job-name=eval-sft-f4-266
#SBATCH --output=logs/eval_sft_fold4_266q_%j.log
#SBATCH --error=logs/eval_sft_fold4_266q_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Evaluate Alex's SFT fold_4 on 266 mixed questions
# This is the correct SFT baseline for exp7/8/9 comparison
# (exp7/8/9 GRPO all start from fold_4, not fold_0)

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_sft_f4_266"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH="$WORKDIR/fine_tuning_qwen/fold_4"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_v4_mixed.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export RUN_NAME="sft_fold4_eval_266q"

if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: SFT fold_4 not found at $LORA_PATH"
    ls -la "$WORKDIR/fine_tuning_qwen/" 2>&1 | grep fold
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_sft_f4_266":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned_only.py"

echo "Job finished at $(date)"
