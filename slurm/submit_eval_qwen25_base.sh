#!/bin/bash
#SBATCH --job-name=eval-qwen25-base
#SBATCH --output=logs/eval_qwen25_base_%j.log
#SBATCH --error=logs/eval_qwen25_base_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Evaluate base Qwen2.5-7B - 190 questions x 4 gen x 5 folds
# Runs McNemar test if fine-tuned results are already available

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_eval_qwen25"
mkdir -p "$WORKDIR/fine_tuning_qwen/logs"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_sft_combined.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_eval_qwen25":/tmp \
  --env HF_HOME="$HF_HOME" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/evaluate_qwen25_base.py"

echo "Job finished at $(date)"
