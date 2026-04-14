# GRPO for Domain-Specific Reliability Engineering

Reinforcement learning (GRPO) approach to improve **Qwen3-8B** on numerical reliability engineering problems. This work investigates whether GRPO can improve domain-specific reasoning without the catastrophic forgetting systematically observed with supervised fine-tuning (SFT).

## Motivation

In prior work ([SFT experiments](https://github.com/helloelora/reliability-engineering-dataset-generator)), all SFT configurations on small expert datasets degraded the base model. The best SFT result on Qwen3-8B (Alex's 18-experiment sweep) was +2.3% on in-distribution data, but not statistically significant (McNemar p=0.5) and consistently caused catastrophic forgetting on held-out questions (up to -20.4pp).

**GRPO hypothesis**: instead of replacing the model's outputs via teacher forcing, reinforce its *correct* reasoning patterns while penalizing errors. The model stays close to its base behavior via KL-constrained policy optimization, avoiding catastrophic forgetting.

## Method

### GRPO with DAPO-style dynamic sampling

We implement a custom GRPO training loop (no TRL dependency) with **dynamic sampling** from the DAPO paper (ByteDance/Tsinghua, 2024):

1. For each training step, sample a question and generate **G=4** completions
2. Score each completion with a rule-based reward:
   - **Correctness** (5% relative tolerance against ground truth)
   - **Format** (presence of `\boxed{}` answer marker)
   - **Partial credit** for close answers
3. If all G generations produce the same reward (all correct or all wrong), **discard and re-sample** a new question - this is the DAPO "dynamic sampling" that ensures every gradient step carries signal
4. Normalize rewards within the group (zero-mean, unit-variance)
5. Update the policy to increase probability of above-average completions

### DeepSeek-R1-style pipeline: SFT then GRPO

Following the DeepSeek-R1 approach, the best configuration uses a **two-stage pipeline**:
1. **SFT warm-start**: load Alex's SFT LoRA (trained on 600 non-mixed questions where the base model scores 0/4 or 4/4)
2. **GRPO**: train on 266 "mixed" questions (where the base model scores 1/4 to 3/4) - these are the questions with actual learning signal

## Datasets

All datasets are derived from textbook problems in reliability engineering (Modarres, Kaminskiy, Krivtsov - *Reliability Engineering and Risk Analysis*), with additional synthetic paraphrases.

| Dataset | Questions | Description | Used by |
|---------|:---------:|-------------|---------|
| `master_dataset_v4` | 866 | Full dataset: 280 base + 586 paraphrases | Source for all splits |
| 266 "mixed" questions | 266 | Pre-screened from v4: base model gets 1/4 to 3/4 correct (has contrastive signal for RL) | GRPO training |
| 600 non-mixed questions | 600 | Base model gets 0/4 or 4/4 correct (no contrastive signal for RL) - used for SFT warm-start only | SFT (Alex) |
| 54 independent holdout | 54 | Never seen during SFT or GRPO training | Out-of-distribution evaluation |

**Lineage**: v2 (280) ⊂ v3 (501) ⊂ v4 (866). Strict subsets, growing only via paraphrases.

**Dataset screening process**: for each candidate question, generate 4 answers with the base model (temperature=0.8), score against ground truth (5% tolerance), keep only "mixed" questions (not 100% correct, not 0% correct). Discard "all correct" (no room to improve) and "all wrong" (no positive signal for RL).

## Key Results

### In-distribution: 266 mixed questions

| Model | Accuracy | Delta vs Base | McNemar p |
|-------|:--------:|:-------------:|:---------:|
| **Qwen3-8B base** | 50.4% (134/266) | - | - |
| SFT fold_4 (Alex) | 50.8% (135/266) | +0.4pp | n.s. |
| GRPO exp10 ckpt-80 (no SFT) | 50.8% (135/266) | +0.4pp | n.s. |
| GRPO exp7 ckpt-200 | 52.3% (139/266) | +1.9pp | - |
| GRPO exp7 ckpt-100 | 54.1% (144/266) | +3.7pp | - |
| **GRPO exp7 ckpt-80** | **57.9% (154/266)** | **+7.5pp** | - |

### Out-of-distribution: 54 independent holdout questions

| Model | Accuracy | Delta vs Base | McNemar p |
|-------|:--------:|:-------------:|:---------:|
| **Qwen3-8B base** | 53.7% (29/54) | - | - |
| SFT fold_4 (Alex) | 33.3% (18/54) | **-20.4pp** | **0.022** |
| GRPO exp7 ckpt-80 | 50.0% (27/54) | -3.7pp | 0.803 |
| GRPO exp10 ckpt-200 (no SFT) | 55.6% (30/54) | +1.9pp | 1.000 |

**Main findings**:
- **GRPO with SFT warm-start achieves +7.5pp** on in-distribution questions (exp7 ckpt-80)
- **GRPO preserves base-model generalization** on holdout - unlike SFT which causes -20.4pp catastrophic forgetting (p=0.022, significant)
- **SFT warm-start is essential**: GRPO from scratch (exp10) barely improves over base (+0.4pp)
- **Early stopping is critical**: performance peaks at ~80 useful steps then declines (57.9% → 52.3% at step 200)

See [RESULTS.md](RESULTS.md) for full experiment details, training dynamics, and analysis.

## GRPO Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Quantization | 4-bit (bitsandbytes) |
| LoRA rank / alpha | 32 / 32 |
| Learning rate | 1e-5 (exp7), 5e-6 (exp9), 2e-5 (exp8) |
| Useful training steps | 200 (with dynamic sampling) |
| Generations per prompt (G) | 4 |
| Gradient accumulation | 4 |
| Max completion length | 3072 tokens |
| Max gradient norm | 0.1 |
| Weight decay | 0.1 |
| Warmup ratio | 0.1 |
| Temperature (training) | 1.0 |
| Top-p | 0.95 |
| Evaluation | Greedy-style (temp=1.0, top_p=0.95, deterministic seed per question) |
| Scoring tolerance | 5% relative |

## Project Structure

```
grpo-reliability-engineering/
├── training/
│   ├── grpo_exp7_dynamic.py       # Main: GRPO with SFT warm-start + dynamic sampling
│   ├── grpo_exp10_nosft.py        # Ablation: GRPO from scratch (no SFT)
│   ├── sft_qwen3_kfold.py         # SFT k-fold cross-validation
│   ├── sft_train_qwen25.py        # SFT on Qwen2.5-7B (baseline)
│   └── dpo_train.py               # DPO alternative approach
├── evaluation/
│   ├── evaluate_single.py         # Evaluate any LoRA on any dataset + McNemar
│   ├── evaluate_finetuned_only.py # Evaluate FT model (base already done separately)
│   ├── evaluate_gsm8k.py          # GSM8K benchmark for general math capability
│   ├── evaluate_qwen25_base.py    # Qwen2.5 baseline with k-fold
│   └── screen_v4.py               # Dataset screening (mixed question selection)
├── generators/
│   ├── augment_grpo_questions.py  # Generate question variations
│   ├── verify_generated.py        # Verify generated answers
│   └── verify_reasoning.py        # Verify reasoning chains
├── slurm/                         # SLURM job scripts for RUCHE HPC
├── results/                       # Evaluation outputs and analysis
├── paper/                         # LaTeX paper and slides
│   ├── paper/main.tex             # Paper source
│   └── slides/slides.tex          # Presentation slides
├── RESULTS.md                     # Detailed experiment results
├── requirements.txt
└── .gitignore
```

## Setup

### RUCHE HPC (recommended)

```bash
# Upload to cluster
scp -r . $USER@ruche:$WORKDIR/fine_tuning_qwen/

# Run GRPO with SFT warm-start (exp7)
sbatch slurm/submit_exp7.sh

# Evaluate checkpoint 80 on 266 mixed questions
sbatch slurm/submit_eval_exp7_ckpt80_266q.sh

# Evaluate on holdout
sbatch slurm/submit_eval_exp7_ckpt80_holdout.sh
```

### Local

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
python training/grpo_exp7_dynamic.py
```

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2402.03300)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (ByteDance/Tsinghua, 2024)
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Unsloth GRPO guide](https://docs.unsloth.ai/basics/reward-training-grpo-and-rl)
- [SFT experiments (prior work)](https://github.com/helloelora/reliability-engineering-dataset-generator)

## License

MIT
