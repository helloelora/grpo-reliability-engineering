#!/bin/bash
#SBATCH --job-name=eval-exp10-holdout
#SBATCH --output=logs/eval_exp10_ckpt200_holdout_%j.log
#SBATCH --error=logs/eval_exp10_ckpt200_holdout_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr
#SBATCH --dependency=afterany:534796

# Evaluate exp10 (GRPO from scratch) checkpoint-200 on 54q independent holdout
# Measures generalization of GRPO without SFT warm-start

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_exp10_holdout"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH="$WORKDIR/fine_tuning_qwen/outputs_exp10/checkpoint-200"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/independent_holdout_54q.jsonl"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export RUN_NAME="exp10_ckpt200_eval_holdout"

if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: checkpoint not found at $LORA_PATH"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: holdout dataset not found at $DATASET_PATH"
    exit 1
fi

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_exp10_holdout":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned_only.py"

echo "Job finished at $(date)"
