#!/bin/bash
#SBATCH --job-name=eval-q3base-266-resume
#SBATCH --output=logs/eval_qwen3_base_266q_resume_%j.log
#SBATCH --error=logs/eval_qwen3_base_266q_resume_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Resume Qwen3-8B BASE eval on 266q — only the last 4 questions (263-266)
# Original job 533785 completed questions 1-262, timed out before finishing

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_q3base_266_resume"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH=""  # empty → evaluate base model with no LoRA
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_v4_mixed.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export RUN_NAME="qwen3_base_eval_266q_resume"
export START_IDX=262   # 0-indexed: questions 1-262 done, resume from index 262
export END_IDX=266

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_q3base_266_resume":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  --env START_IDX="$START_IDX" \
  --env END_IDX="$END_IDX" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned_only_resume.py"

echo "Job finished at $(date)"
