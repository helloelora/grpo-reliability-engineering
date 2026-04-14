#!/bin/bash
#SBATCH --job-name=eval-exp7-80
#SBATCH --output=logs/eval_exp7_ckpt80_266q_%j.log
#SBATCH --error=logs/eval_exp7_ckpt80_266q_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Evaluate exp7 checkpoint-80 (training peak observed at steps 76-100)
# on the 266 mixed questions (training set, in-distribution measure)

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_exp7_80"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export LORA_PATH="$WORKDIR/fine_tuning_qwen/outputs_exp7/checkpoint-80"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_v4_mixed.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export RUN_NAME="exp7_ckpt80_eval_266q"

if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: checkpoint not found at $LORA_PATH"
    exit 1
fi

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_exp7_80":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env LORA_PATH="$LORA_PATH" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_finetuned_only.py"

echo "Job finished at $(date)"
