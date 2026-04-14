#!/bin/bash
#SBATCH --job-name=grpo-exp10
#SBATCH --output=logs/grpo_exp10_%j.log
#SBATCH --error=logs/grpo_exp10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Exp 10: GRPO from scratch (NO SFT warm-start)
# Same lr as exp7 (1e-5), same dataset, same hyperparameters
# Only difference: fresh LoRA instead of loading Alex's SFT fold_4
# Ablation to test if SFT warm-start helps GRPO

echo "Job started at $(date)"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE SCRATCH SCRATCH_DIR TMPDIR

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_exp10"
mkdir -p "$WORKDIR/torch_inductor_cache_exp10"
mkdir -p "$WORKDIR/triton_cache_exp10"
mkdir -p "$WORKDIR/scratch_exp10"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_v4_mixed.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen/outputs_exp10"
export LEARNING_RATE="1e-5"
export MAX_USEFUL_STEPS="200"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_exp10":/tmp \
  --bind "$WORKDIR/scratch_exp10":/scratch \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=0 \
  --env TRANSFORMERS_OFFLINE=0 \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env LEARNING_RATE="$LEARNING_RATE" \
  --env MAX_USEFUL_STEPS="$MAX_USEFUL_STEPS" \
  --env TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_inductor_cache_exp10" \
  --env TRITON_CACHE_DIR="$WORKDIR/triton_cache_exp10" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TMPDIR=/tmp \
  --env TMP=/tmp \
  --env TEMP=/tmp \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/grpo_exp10_nosft.py"

echo "Job finished at $(date)"
