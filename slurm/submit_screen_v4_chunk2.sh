#!/bin/bash
#SBATCH --job-name=screen-c2
#SBATCH --output=logs/screen_v4_chunk2_%j.log
#SBATCH --error=logs/screen_v4_chunk2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=23:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elora.drouilhet@student-cs.fr

# Screen v4 dataset chunk 2 of 3
# Generate 4 base model answers per question, identify mixed (1/4, 2/4, 3/4)

echo "Job started at $(date)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$WORKDIR/mytmp_screen_c2"

export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export HF_HOME="$WORKDIR/.cache/huggingface"
export DATASET_PATH="$WORKDIR/fine_tuning_qwen/master_dataset_v4.jsonl"
export OUTPUT_DIR="$WORKDIR/fine_tuning_qwen"
export CHUNK_INDEX=2
export N_CHUNKS=3

apptainer exec   --nv   --writable-tmpfs   --bind "$WORKDIR":"$WORKDIR":rw   --bind "$WORKDIR/mytmp_screen_c2":/tmp   --env HF_HOME="$HF_HOME"   --env DATASET_PATH="$DATASET_PATH"   --env OUTPUT_DIR="$OUTPUT_DIR"   --env CHUNK_INDEX=$CHUNK_INDEX   --env N_CHUNKS=$N_CHUNKS   "$WORKDIR/unsloth_latest.sif"   python "$WORKDIR/fine_tuning_qwen/screen_v4.py"

echo "Job finished at $(date)"
