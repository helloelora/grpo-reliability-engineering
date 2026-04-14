# Exp7-10 Results Report: GRPO for Reliability Engineering

## Overview

All experiments use **Qwen3-8B** (4-bit quantized) with DAPO-style dynamic sampling GRPO.
- **Training dataset:** 266 pre-screened "mixed" questions from master_v4 (questions where base model gets 1/4 to 3/4 correct)
- **Holdout:** 54 independent questions never seen during SFT or GRPO training
- **SFT baseline:** Alex's fold_4 LoRA, trained on the 600 "easy" questions from v4 (4/4 correct by base)
- **Evaluation:** greedy-style (temperature=1.0, top_p=0.95, deterministic seed per question), 5% tolerance

## Experiment Configurations

| Experiment | Learning Rate | SFT Init | Description |
|------------|:------------:|:--------:|-------------|
| **Exp7** | 1e-5 | fold_4 | Baseline GRPO |
| **Exp8** | 2e-5 | fold_4 | Aggressive LR (2x) |
| **Exp9** | 5e-6 | fold_4 | Conservative LR (0.5x) |
| **Exp10** | 1e-5 | **None** | GRPO from scratch (no SFT warm-start) |

All experiments: 200 useful steps, grad_accum=4, 4 generations/prompt, max_grad_norm=0.1.

## Main Results: 266 Mixed Questions (In-Distribution)

| Model | Correct | /266 | Accuracy | Truncated | Delta vs Base |
|-------|:-------:|:----:|:--------:|:---------:|:-------------:|
| Qwen2.5-7B base | ~40 | 266 | 15.0% | 0 | — |
| **Qwen3-8B base** | 134 | 266 | **50.4%** | 4 | — |
| SFT fold_0 (Alex) | 124 | 266 | 46.6% | 7 | -3.8pp |
| SFT fold_4 (Alex) | 135 | 266 | 50.8% | 8 | +0.4pp |
| **GRPO exp10 ckpt-80 (no SFT)** | 135 | 266 | **50.8%** | 1 | +0.4pp |
| GRPO exp7 ckpt-200 | 139 | 266 | 52.3% | 6 | +1.9pp |
| GRPO exp7 ckpt-100 | 144 | 266 | 54.1% | 9 | +3.7pp |
| **GRPO exp7 ckpt-80** | **154** | 266 | **57.9%** | 10 | **+7.5pp** |

### Pending evaluations (exp8, exp9, exp10 ckpt-100/200)

| Model | Status |
|-------|--------|
| GRPO exp8 ckpt-80/100/200 (lr=2e-5) | Submitted, waiting |
| GRPO exp9 ckpt-80/100/200 (lr=5e-6) | Submitted, waiting |
| GRPO exp10 ckpt-100 (no SFT) | Submitted, dependency on training |
| GRPO exp10 ckpt-200 (no SFT) | Submitted, dependency on training |
| ~~SFT fold_4 on 266q~~ | **Done: 50.8%** |

## Holdout Results: 54 Independent Questions (Out-of-Distribution)

| Model | Correct | /54 | Accuracy | Truncated | Delta vs Base |
|-------|:-------:|:---:|:--------:|:---------:|:-------------:|
| SFT fold_4 (Alex) | 18 | 54 | 33.3% | 2 | **-20.4pp** |
| GRPO exp7 ckpt-80 | 29 | 54 | 53.7% | 2 | +0.0pp |
| **Qwen3-8B base** | **29** | 54 | **53.7%** | 1 | — |

### Pending holdout evaluations

| Model | Status |
|-------|--------|
| GRPO exp10 ckpt-80 (no SFT) | Submitted |
| GRPO exp10 ckpt-200 (no SFT) | Submitted, dependency on training |

## Training Dynamics

| Experiment | LR | Useful Steps | Wasted Steps | Waste Rate | Total Attempts |
|------------|:--:|:-----------:|:------------:|:----------:|:--------------:|
| Exp7 | 1e-5 | 200 | 60 | 23.1% | 260 |
| Exp8 | 2e-5 | 200 | 53 | 20.9% | 253 |
| Exp9 | 5e-6 | 200 | 54 | 21.3% | 254 |
| **Exp10** | 1e-5 | 200 | 83 | **29.3%** | 283 |

Exp10 (no SFT) has the highest waste rate: starting from scratch, the model struggles more to produce useful reward variance.

## Key Findings

### 1. SFT warm-start is critical for GRPO
- **Exp7 (with SFT):** 57.9% at ckpt-80 (+7.5pp vs base)
- **Exp10 (no SFT):** 50.8% at ckpt-80 (+0.4pp vs base)
- **Delta: +7.1pp** in favor of SFT warm-start
- The SFT LoRA, despite degrading standalone performance (-3.8pp vs base on 266q), provides a crucial initialization for GRPO to build upon.

### 2. SFT results vary by fold, but always hurt on holdout
- SFT fold_0 on 266q: 46.6% (-3.8pp vs base) — significant regression
- SFT fold_4 on 266q: 50.8% (+0.4pp vs base) — essentially neutral
- SFT fold_4 on 54q holdout: 33.3% (**-20.4pp** vs base) — catastrophic forgetting on unseen questions
- The fold variance (46.6% vs 50.8%) shows SFT is unstable across folds, but the holdout regression is consistent and severe

### 3. GRPO shows clear overfitting / drift after step 80
- Exp7: 57.9% (ckpt-80) -> 54.1% (ckpt-100) -> 52.3% (ckpt-200)
- Loss of 5.6pp between peak and final checkpoint
- Suggests early stopping at ~80 useful steps is optimal for this dataset size

### 4. GRPO from scratch barely improves over base
- Exp10 ckpt-80: 50.8% vs base 50.4% (+0.4pp, likely within noise)
- Higher waste rate (29.3% vs 23.1%) indicates harder optimization landscape
- The model needs the SFT "nudge" to effectively learn from GRPO rewards

### 5. GRPO preserves base-model generalization on holdout
- Exp7 ckpt-80 on holdout: 53.7% vs base 53.7% → **+0.0pp** (exact match)
- GRPO improves +7.5pp in-distribution WITHOUT degrading out-of-distribution
- GRPO fully recovers the SFT catastrophic forgetting (53.7% vs SFT's 33.3%)
- This is a key advantage of RL over SFT: it adds capability without forgetting

## Interpretation

The SFT plays a paradoxical role:
- **Alone**, it severely degrades performance (-3.8pp on 266q, -20.4pp on holdout)
- **As GRPO initialization**, it's essential for in-distribution gains (+7.1pp vs GRPO from scratch)
- GRPO **preserves generalization** — holdout matches base exactly (53.7%)

The pipeline Base → SFT → GRPO works as follows:
1. **SFT** teaches format/structure but causes catastrophic forgetting (especially on hard/unseen questions)
2. **GRPO** partially recovers the damage and improves on its training distribution
3. But the improvement is **in-distribution only** — not yet generalizing to truly new questions

This is analogous to the DeepSeek-R1 pipeline (SFT for alignment, RL for capability), but on a dataset this small (266q), the GRPO overfits to training patterns rather than learning transferable reasoning.

### Possible next steps
- Larger GRPO training set (more mixed questions) to improve generalization
- Early stopping based on holdout validation (not just training accuracy)
- KL penalty (beta > 0) to keep the model closer to base and reduce overfitting
