#!/bin/bash
#SBATCH --job-name=eval-dpo
#SBATCH --output=logs/eval_dpo_%j.log
#SBATCH --error=logs/eval_dpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Evaluate DPO fine-tuned model on 3 splits

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_dpo"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH="$WORKDIR/fine_tuning_qwen/lora_model_dpo"
export DATA_DIR="$WORKDIR/fine_tuning_qwen"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_dpo":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATA_DIR="$DATA_DIR" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned.py"

echo "Job finished at $(date)"
