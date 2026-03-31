#!/bin/bash
#SBATCH --job-name=eval-finetuned
#SBATCH --output=logs/eval_finetuned_%j.log
#SBATCH --error=logs/eval_finetuned_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00

# ============================================================
# Evaluate fine-tuned model on 3 splits (255 questions x 4 gen)
# Run in parallel with submit_eval_base.sh
# ============================================================

echo "=== Eval V3 FINETUNED started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_ft"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH="$WORKDIR/fine_tuning_qwen/lora_model"
export DATA_DIR="$WORKDIR/fine_tuning_qwen"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_ft":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATA_DIR="$DATA_DIR" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned.py"

echo "=== Eval V3 FINETUNED finished at $(date) ==="
