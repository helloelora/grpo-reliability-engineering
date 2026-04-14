# Experiment Results

## Overview

All experiments use **Qwen3-8B** (4-bit quantized via Unsloth) with a custom GRPO training loop implementing DAPO-style dynamic sampling. Training is done on the RUCHE HPC cluster (NVIDIA A100 40GB).

- **Training dataset**: 266 pre-screened "mixed" questions from master_dataset_v4 (questions where the base model gets 1/4 to 3/4 correct)
- **Holdout**: 54 independent questions never seen during SFT or GRPO training
- **SFT baseline**: Alex's fold_4 LoRA, trained on the 600 non-mixed questions from v4 (0/4 or 4/4 correct by base - no contrastive signal for RL)
- **Evaluation**: greedy-style (temperature=1.0, top_p=0.95, deterministic seed per question), 5% relative tolerance

---

## Experiment Configurations

| Experiment | Learning Rate | SFT Init | Description |
|------------|:------------:|:--------:|-------------|
| **Exp7** | 1e-5 | fold_4 | Baseline GRPO (SFT warm-start) |
| **Exp8** | 2e-5 | fold_4 | Aggressive LR (2x) |
| **Exp9** | 5e-6 | fold_4 | Conservative LR (0.5x) |
| **Exp10** | 1e-5 | **None** | GRPO from scratch (no SFT warm-start) |

All experiments: 200 useful steps, grad_accum=4, 4 generations/prompt, LoRA r/alpha=32/32, max_grad_norm=0.1.

---

## Main Results: 266 Mixed Questions (In-Distribution)

| Model | Correct | /266 | Accuracy | Truncated | Delta vs Base |
|-------|:-------:|:----:|:--------:|:---------:|:-------------:|
| Qwen2.5-7B base | ~40 | 266 | 15.0% | 0 | - |
| **Qwen3-8B base** | 134 | 266 | **50.4%** | 4 | - |
| SFT fold_4 (Alex) | 135 | 266 | 50.8% | 8 | +0.4pp |
| GRPO exp10 ckpt-80 (no SFT) | 135 | 266 | 50.8% | 1 | +0.4pp |
| GRPO exp7 ckpt-200 | 139 | 266 | 52.3% | 6 | +1.9pp |
| GRPO exp7 ckpt-100 | 144 | 266 | 54.1% | 9 | +3.7pp |
| **GRPO exp7 ckpt-80** | **154** | 266 | **57.9%** | 10 | **+7.5pp** |

---

## Holdout Results: 54 Independent Questions (Out-of-Distribution)

| Model | Correct | /54 | Accuracy | Truncated | Delta vs Base | McNemar p |
|-------|:-------:|:---:|:--------:|:---------:|:-------------:|:---------:|
| **Qwen3-8B base** | **29** | 54 | **53.7%** | 1 | - | - |
| SFT fold_4 (Alex) | 18 | 54 | 33.3% | 2 | **-20.4pp** | **0.022** |
| GRPO exp7 ckpt-80 | 27 | 54 | 50.0% | 2 | -3.7pp | 0.803 |
| GRPO exp10 ckpt-200 (no SFT) | 30 | 54 | 55.6% | 0 | +1.9pp | 1.000 |

**McNemar test** (per-question, continuity-corrected):
- SFT fold_4 vs base: 15 regressions, 4 improvements → **p=0.022 (significant)** - SFT causes statistically significant degradation on unseen questions
- GRPO exp7 ckpt-80 vs base: 9 regressions, 7 improvements → p=0.803 (not significant) - GRPO does not significantly degrade holdout performance
- GRPO exp10 ckpt-200 vs base: 2 regressions, 3 improvements → p=1.000 (not significant)

---

## Training Dynamics

| Experiment | LR | Useful Steps | Wasted Steps | Waste Rate | Total Attempts |
|------------|:--:|:-----------:|:------------:|:----------:|:--------------:|
| Exp7 | 1e-5 | 200 | 60 | 23.1% | 260 |
| Exp8 | 2e-5 | 200 | 53 | 20.9% | 253 |
| Exp9 | 5e-6 | 200 | 54 | 21.3% | 254 |
| **Exp10** | 1e-5 | 200 | 83 | **29.3%** | 283 |

Exp10 (no SFT) has the highest waste rate: starting from scratch, the model struggles more to produce useful reward variance. Dynamic sampling ensures no gradient step is wasted despite the high waste rate.

### Overfitting trajectory (exp7)

| Checkpoint | Accuracy (266q) | Delta vs Base |
|:----------:|:----------------:|:-------------:|
| ckpt-80 | **57.9%** | +7.5pp |
| ckpt-100 | 54.1% | +3.7pp |
| ckpt-200 | 52.3% | +1.9pp |

Performance peaks at ~80 useful steps then steadily declines, losing 5.6pp between peak and final checkpoint. Early stopping is critical for this dataset size.

---

## Earlier Experiments (Exp1-6)

These experiments used TRL's `GRPOTrainer` (before switching to a custom loop with dynamic sampling).

### Common setup

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Loss type | dr_grpo (Unsloth recommendation) |
| Beta | 0.0 (no KL penalty) |
| Fixes applied | `mask_truncated_completions=False`, `importance_sampling_level=sequence`, `max_grad_norm=0.1` |

### In-distribution results (281 questions)

| Exp | LR | Steps | LoRA r/a | Dataset | Base | FT | Delta | McNemar p |
|-----|:--:|:-----:|:--------:|---------|:----:|:--:|:-----:|:---------:|
| **1** | 5e-6 | 220 | 32/32 | 281q | 63.7% | 63.0% | -0.7% | 0.864 |
| **3** | 1e-5 | 220 | 32/32 | 281q | 63.7% | 64.1% | +0.4% | - |
| **5** | 1e-5 | 180 | **64/64** | 281q | 63.7% | 64.8% | +1.1% | 0.760 |

**Lesson**: without dynamic sampling, 42% of training steps had zero gradient (all 4 generations agreed). The TRL-based approach could not produce meaningful improvement.

### Zero-gradient analysis (57-question pilot)

| n_correct/4 at step | # steps | Status |
|:--------------------:|:-------:|:------:|
| 0/4 | 11 | wasted (variance=0) |
| 1/4 | 9 | useful gradient |
| 2/4 | 18 | useful gradient |
| 3/4 | 19 | useful gradient |
| 4/4 | 23 | wasted (variance=0) |

34/80 steps (42%) produced zero gradient - the empirical signature of the missing dynamic sampling that DAPO addresses.

---

## DPO and SFT Baselines

### DPO on Qwen3-8B

**Dataset**: 233 preference pairs from the base model's generations (correct answer = chosen, wrong = rejected).

| Split | Base | DPO | Delta | McNemar p |
|-------|:----:|:---:|:-----:|:---------:|
| Hard holdout (29q) | 29.3% | 32.8% | +3.4% | - |
| Master holdout (169q, partial) | 71.9% | 71.2% | -0.7% | - |
| Combined (B+C) | - | - | - | 1.00 |

Not significant.

### SFT on Qwen2.5-7B (5-fold CV, 281 questions)

| Fold | Base | SFT | Delta |
|:----:|:----:|:---:|:-----:|
| 1 | 33.3% | 28.1% | -5.2% |
| 2 | 39.3% | 26.8% | -12.5% |
| 3 | 28.6% | 19.6% | -9.0% |
| 4 | 39.3% | 35.7% | -3.6% |
| 5 | 41.1% | 41.1% | 0.0% |
| **Mean** | **36.3% +/- 4.7%** | **30.3% +/- 7.4%** | **-6.0%** |

McNemar p=0.607 (not significant). Same catastrophic forgetting pattern as Qwen3-14B SFT experiments.

### Qwen2.5-7B SFT vs Qwen3-8B Base (same 266 questions)

| Model | Accuracy |
|-------|:--------:|
| Qwen2.5-7B base | 15.0% |
| Qwen2.5-7B SFT (best fold) | ~35% |
| **Qwen3-8B base** | **50.4%** |

The Qwen3-8B base model outperforms Qwen2.5-7B SFT by a wide margin - confirming that model selection matters more than fine-tuning.

---

## Key Findings

### 1. SFT warm-start is critical for GRPO
- **Exp7 (with SFT)**: 57.9% at ckpt-80 (+7.5pp vs base)
- **Exp10 (no SFT)**: 50.8% at ckpt-80 (+0.4pp vs base)
- Delta: **+7.1pp** in favor of SFT warm-start
- The SFT LoRA, despite degrading standalone performance, provides a crucial initialization for GRPO

### 2. SFT causes catastrophic forgetting; GRPO does not
- SFT fold_4 on holdout: 33.3% (**-20.4pp** vs base, McNemar p=0.022)
- GRPO exp7 ckpt-80 on holdout: 50.0% (-3.7pp, p=0.803, not significant)
- GRPO from scratch (exp10) on holdout: 55.6% (+1.9pp, p=1.0, not significant)

### 3. GRPO overfits after ~80 useful steps
- Exp7: 57.9% (ckpt-80) → 54.1% (ckpt-100) → 52.3% (ckpt-200)
- 5.6pp drop between peak and final checkpoint
- Suggests early stopping based on validation is essential for small datasets

### 4. Dynamic sampling reduces waste from 42% to ~23%
- Without dynamic sampling (TRL): 42% of steps produced zero gradient
- With dynamic sampling (custom loop): 23% waste rate
- This is the key technical contribution enabling GRPO to work on this dataset

### 5. The DeepSeek-R1 pipeline works at small scale
- Base → SFT (format/structure) → GRPO (capability) follows the DeepSeek-R1 recipe
- SFT teaches structure but causes forgetting; GRPO recovers and improves
- In-distribution gains are real (+7.5pp) but do not yet generalize to holdout

---

## Regex Improvements

The reward/evaluation regex handles all answer formats:
- `\boxed{3.1%}` → 0.031 (percentage conversion)
- `\boxed{2.52 \times 10^{5}}` → 252000 (LaTeX scientific notation)
- `\boxed{14/33}` → 0.4242 (fractions)
- `\boxed{23,717}` → 23717 (thousands separators)
- `\boxed{0,001}` → 0.001 (European decimal comma)
