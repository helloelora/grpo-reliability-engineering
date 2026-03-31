# GRPO for reliability engineering — Qwen3-8B

Reinforcement learning (GRPO) approach to improve Qwen3-8B on reliability engineering problems. This work investigates whether GRPO can improve domain-specific numerical reasoning without catastrophic forgetting, using a curated dataset of questions filtered by answer diversity.

## Approach

### 1. Dataset curation via answer diversity screening

Instead of training on all available questions, we filter for questions where the base model shows **mixed performance** — sometimes correct, sometimes wrong. This ensures the RL training has contrastive signal (both positive and negative examples from the model's own generations).

**Screening process:**
1. For each candidate question, generate 4 answers with the base model (temperature=0.8)
2. Score each answer against ground truth (5% relative tolerance)
3. Keep only questions with **mixed results** (not 100% correct, not 0% correct)
4. Discard "all correct" (no room to improve) and "all wrong" (no positive signal for RL)

**Source datasets screened:**

| Source | Total | All correct | Mixed (kept) | All wrong |
|--------|-------|-------------|--------------|-----------|
| `master_dataset_cleaned_numeric` | 80 | 40 | 21 | 19 |
| `hard_numeric_generated` | 65 | 7 | 36 | 22 |
| **Combined training set** | | | **57** | |

### 2. GRPO training

**Group relative policy optimization** generates multiple completions per question, scores them with rule-based rewards, and reinforces above-average completions while staying close to the base model via KL penalty.

**Reward function** extracts the last `\boxed{}` value and compares to ground truth:

| Relative error | Reward |
|----------------|--------|
| ≤ 0.1% | 1.0 (exact) |
| ≤ 1% | 0.8 (very close) |
| ≤ 5% | 0.4 (close) |
| > 5% | 0.01 (wrong but formatted) |
| No `\boxed{}` | 0.0 |
| > 2 `\boxed{}` | -0.3 (hedging penalty) |

The numeric extraction regex handles multiple answer formats transparently:
- Percentages: `\boxed{3.1\%}` → 0.031
- LaTeX scientific notation: `\boxed{2.52 \times 10^{5}}` → 252000
- Fractions: `\boxed{14/33}` → 0.4242
- Thousands separators: `\boxed{23,717}` → 23717
- European decimal comma: `\boxed{0,001}` → 0.001
- Standard e-notation: `\boxed{1.5e-3}` → 0.0015

### 3. Evaluation protocol

Three evaluation splits, each tested with 4 generations per question on both base and fine-tuned models:

| Split | Questions | Purpose |
|-------|-----------|---------|
| A. Training set | 57 | Did GRPO improve on these problems? |
| B. Hard holdout | 29 | Generalization to unseen hard problems |
| C. Master holdout | 169 | Catastrophic forgetting check |

## Configuration

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| LoRA rank / alpha | 32 / 32 |
| Generations per prompt | 4 |
| KL coefficient (β) | 0.001 |
| Loss type | DAPO |
| Sampling temperature | 0.8 |
| Max completion length | 6144 tokens |
| Training steps | 80 |
| Learning rate | 5e-6 |
| Thinking mode | Disabled (`/no_think`) |
| Hardware | NVIDIA A100-SXM4-40GB (RUCHE HPC) |

## Results

| Split | Base model | Fine-tuned | Delta |
|-------|------------|------------|-------|
| A. Training (57q) | 58.3% (133/228) | 57.9% (132/228) | **-0.4%** |
| B. Hard holdout (29q) | 29.3% (34/116) | 32.8% (38/116) | **+3.4%** |
| C. Master holdout (169q) | 71.9% (486/676) | 71.2% (481/676) | **-0.7%** |

Per-question movement:

| Split | Improved | Degraded | Unchanged |
|-------|----------|----------|-----------|
| A. Training | 20 | 16 | 21 |
| B. Hard holdout | 7 | 5 | 17 |
| C. Master holdout | 19 | 23 | 127 |

**Conclusion:** no statistically significant improvement. The deltas are within sampling noise (4 generations at temperature 0.8). No catastrophic forgetting was observed either — the fine-tuned model performs equivalently to the base model across all splits.

This is consistent with [parallel SFT experiments](https://github.com/ADnocap/Reliability-Domain-Specific-LLM) by a colleague on the same model and domain, where 17 SFT experiments also failed to achieve statistically significant improvement (best: +2.3%, p > 0.05).

### Interpretation

The base Qwen3-8B already achieves ~72% on reliability engineering problems. With only 57 training questions, neither GRPO nor SFT provides enough signal to meaningfully shift model behavior. The "self-instruct ceiling effect" (synthetic questions being easier than real textbook problems) further limits the usefulness of generated training data.

## Project structure

```
grpo-reliability-engineering/
├── training/
│   └── grpo_train.py                    # GRPO training script
├── evaluation/
│   ├── evaluate_base.py                 # Base model evaluation
│   └── evaluate_finetuned.py            # Fine-tuned model evaluation
├── slurm/
│   ├── submit_grpo.sh                   # GRPO training job
│   ├── submit_eval_base.sh              # Base eval job
│   └── submit_eval_finetuned.sh         # Fine-tuned eval job
├── results/
│   └── Last run/                        # Evaluation summaries
│       ├── eval_base_summary.json
│       └── eval_finetuned_summary.json
├── README.md
└── requirements.txt
```

Data files are kept locally (not in the repository). See the datasets section below.

## Datasets

All datasets are stored locally in `data/` (gitignored).

| File | Questions | Description | Origin |
|------|-----------|-------------|--------|
| `master_dataset_cleaned_numeric.json` | 190 | Full numeric dataset (single target per question) | Textbook + synthetic + cross-model verified |
| `hard_numeric_generated.jsonl` | 65 | Hard synthetic questions | Generated by colleague's pipeline |
| `dataset_grpo_combined.json` | 57 | **GRPO training set** (21 + 36 filtered by diversity) | Screened from above two |
| `eval_A_training.json` | 57 | Eval split A — same as training set | = `dataset_grpo_combined.json` |
| `eval_B_hard_holdout.json` | 29 | Eval split B — hard questions excluded from training | `hard_numeric_generated` \ training |
| `eval_C_master_holdout.json` | 169 | Eval split C — master questions excluded from training | `master_dataset_cleaned_numeric` \ training |

## Usage

```bash
# Copy files to HPC
scp training/grpo_train.py $USER@ruche:$WORKDIR/fine_tuning_qwen/
scp data/dataset_grpo_combined.json $USER@ruche:$WORKDIR/fine_tuning_qwen/

# Train
sbatch slurm/submit_grpo.sh

# Evaluate (run both in parallel on 2 GPUs)
sbatch slurm/submit_eval_base.sh
sbatch slurm/submit_eval_finetuned.sh
```

## References

- [DeepSeek-R1: Incentivizing reasoning capability in LLMs via RL](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Parallel SFT experiments (colleague)](https://github.com/ADnocap/Reliability-Domain-Specific-LLM)
- Textbook: Modarres, Kaminskiy, Krivtsov — *Reliability engineering and risk analysis*

## License

MIT
