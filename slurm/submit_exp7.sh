#!/bin/bash
#SBATCH --job-name=grpo-exp7
#SBATCH --output=logs/grpo_exp7_%j.log
#SBATCH --error=logs/grpo_exp7_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Exp 7: Alex's NEW SFT (fold_0) + GRPO on 266 pre-screened mixed questions
# DeepSeek-R1 cold-start pipeline with best hyperparameters from exp1-6bis
#
# Key improvements vs exp6bis:
#   - Custom GRPO loop (no TRL) with DAPO-style dynamic sampling
#   - Dataset pre-screened (strict mixed 1/4-3/4) → lower waste rate
#   - No overlap between SFT data (600q) and GRPO data (266q)
#   - lr=1e-5 (2x, validated in exp3/exp5)
#   - 150 USEFUL steps target (wasted steps don't count)
#   - Checkpoints every 20 useful steps for resume
#   - Full training stats logged (useful/wasted/resample counts)

# >>> EDIT IF NEEDED <<<
FOLD=4  # Best performing fold (78.2% accuracy)

echo "Job started at $(date)"

# Ensure HuggingFace online mode (Unsloth needs it for model loading)
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_exp7"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export FOLD="$FOLD"
export SFT_LORA_PATH="$WORKDIR/fine_tuning_qwen/fold_${FOLD}"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/dataset_grpo_v4_mixed.json"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen/outputs_exp7"

if [ ! -d "$SFT_LORA_PATH" ]; then
    echo "ERROR: SFT LoRA not found at $SFT_LORA_PATH"
    echo "Available folds:"
    ls -la "$WORKDIR/fine_tuning_qwen/adapters/" 2>&1
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

mkdir -p "$WORKDIR/torch_inductor_cache_exp7"
mkdir -p "$WORKDIR/triton_cache_exp7"
mkdir -p "$WORKDIR/scratch_exp7"

unset SCRATCH SCRATCH_DIR TMPDIR

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$WORKDIR":"$WORKDIR":rw \
  --bind "$WORKDIR/mytmp_exp7":/tmp \
  --bind "$WORKDIR/scratch_exp7":/scratch \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=0 \
  --env TRANSFORMERS_OFFLINE=0 \
  --env FOLD="$FOLD" \
  --env SFT_LORA_PATH="$SFT_LORA_PATH" \
  --env DATASET_PATH="$DATASET_PATH" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env TORCHINDUCTOR_CACHE_DIR="$WORKDIR/torch_inductor_cache_exp7" \
  --env TRITON_CACHE_DIR="$WORKDIR/triton_cache_exp7" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHDYNAMO_DISABLE=1 \
  --env TORCH_COMPILE_DISABLE=1 \
  --env UNSLOTH_COMPILE_DISABLE=1 \
  --env TMPDIR=/tmp \
  --env TMP=/tmp \
  --env TEMP=/tmp \
  --env TORCHDYNAMO_DISABLE=1 \
  --env TORCH_COMPILE_DISABLE=1 \
  --env UNSLOTH_COMPILE_DISABLE=1 \
  "$WORKDIR/unsloth_latest.sif" \
  python "$WORKDIR/fine_tuning_qwen/grpo_exp7_dynamic.py"

echo "Job finished at $(date)"
