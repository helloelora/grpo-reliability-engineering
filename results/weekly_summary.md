# Evaluation splits (used for GRPO and DPO)

For the Qwen3-8B experiments (GRPO and DPO), we evaluated on 2 held-out splits (questions never seen during training):

| Split | Questions | Source | Purpose |
|-------|-----------|--------|---------|
| B. Hard holdout | 29 | questions from `hard_numeric_generated.jsonl` excluded from training (all-correct or all-wrong during screening) | generalization to unseen hard problems |
| C. Master holdout | 169 | questions from `master_dataset_cleaned_numeric.json` excluded from training | catastrophic forgetting check |

For the Qwen2.5-7B SFT experiment, we used 5-fold cross-validation on 281 questions instead (~225 train / ~56 test per fold).

---

## 1. GRPO on Qwen3-8B (no-think)

**Goal:** use reinforcement learning to improve the model without catastrophic forgetting.

**Dataset curation:**
- Started from `master_dataset_cleaned_numeric.json` (190 questions, single numeric answer)
- Generated 4 answers per question with the base model (temperature=0.8)
- Kept only questions where the model had a mix of correct and wrong answers (not 100% correct, not 0% correct) — this ensures the RL has contrastive signal
- Result: 21 questions from master dataset
- Same screening on `hard_numeric_generated.jsonl` (65 hard synthetic questions from Alex) → 36 kept
- Final training set: `dataset_grpo_combined.json` — **57 questions**

**Training:** Qwen3-8B, LoRA r=32, 80 steps, DAPO loss, beta=0.001, temperature=0.8

**Evaluation:** 2 held-out splits, 4 generations per question, regex-based scoring (5% tolerance)

| Split | Base | GRPO | Delta | Improved | Degraded | Same |
|-------|------|------|-------|----------|----------|------|
| B. Hard holdout (29q) | 29.3% | 32.8% | +3.4% | 7 | 5 | 17 |
| C. Master holdout (169q) | 71.9% | 71.2% | -0.7% | 19 | 23 | 127 |

McNemar test (on B+C, per generation): base correct → GRPO wrong = 46, base wrong → GRPO correct = 45. p = 1.00 — not significant.

**Example of improvement (split B):** fleet engine failure analysis, target=0.06516
- Base: 0/4 correct, predictions=[0.021, 0.014, 0.023, 0.025]
- GRPO: 3/4 correct, predictions=[0.064, 0.024, 0.066, 0.066]

**Example of regression (split C):** Bayes theorem with 3 assembly plants, target=0.2841
- Base: 4/4 correct, predictions=[0.2841, 0.2841, 0.2841, 0.2841]
- GRPO: 3/4 correct, predictions=[0.2841, 0.00284, 0.2841, 0.2841] — one generation off by a factor 100

---

## 2. DPO on Qwen3-8B (no-think)

**Goal:** use the existing generations as preference pairs (correct answer = chosen, wrong answer = rejected) instead of RL exploration.

**Dataset:** `dataset_dpo_pairs.json` — **233 preference pairs** extracted from the base model's 4 generations on all 255 questions (any question with at least 1 correct + 1 wrong answer gives pairs).

**Training:** Qwen3-8B, LoRA r=32, 3 epochs, beta=0.1, lr=5e-6

**Evaluation:** same 2 held-out splits as GRPO, same regex scoring

| Split | Base | DPO | Delta | Improved | Degraded | Same |
|-------|------|-----|-------|----------|----------|------|
| B. Hard holdout (29q) | 29.3% | 32.8% | +3.4% | 7 | 6 | 16 |
| C. Master holdout (12/169, partial) | 70.8% | 66.7% | -4.2% | 0 | 1 | 11 |

McNemar test (on B+C, per generation): base correct → DPO wrong = 9, base wrong → DPO correct = 11. p = 0.82 — not significant.

Note: split C is partial (12/169 questions) because the evaluation job was killed before completing.

---

## 3. SFT on Qwen2.5-7B (non-reasoning model)

**Goal:** test if a non-reasoning model (no thinking mode) benefits more from fine-tuning, since it has a lower baseline and more room to improve.

**Dataset:** `dataset_sft_combined.json` — **281 questions** (190 from master + 65 hard generated + 26 additional single-answer questions found in other dataset files). All answers normalized to plain numbers.

**Training:** Qwen2.5-7B-Instruct, LoRA r=16, 2 epochs, lr=2e-4, adamw_8bit, cosine scheduler, 5-fold cross-validation (~225 train / ~56 test per fold)

**Evaluation:** 1 generation per question (temperature=0.6, fixed seed per question for reproducibility), regex-based scoring

| Fold | Base | SFT | Delta |
|------|------|-----|-------|
| 1 | 33.3% | 28.1% | -5.2% |
| 2 | 39.3% | 26.8% | -12.5% |
| 3 | 28.6% | 19.6% | -9.0% |
| 4 | 39.3% | 35.7% | -3.6% |
| 5 | 41.1% | 41.1% | 0.0% |
| **Mean** | **36.3% +/- 4.7%** | **30.3% +/- 7.4%** | **-6.0%** |

McNemar test: p = 0.607 — not significant. Base correct → SFT wrong: 9 questions. Base wrong → SFT correct: 6 questions.

**Example of improvement (fold 1):** system reliability problem, target=0.9874
- Base: wrong, predicted 0.1112
- SFT: correct, predicted 0.9713

**Example of regression (fold 1):** redundant power supply system, target=0.9955
- Base: correct
- SFT: wrong, predicted 0.924

---

## Key takeaways

- **No method produced statistically significant improvement** (GRPO p=1.00, DPO p=0.82, SFT p=0.61)
- Qwen3-8B base is already strong (72% on master dataset) — hard to improve with small data
- Qwen2.5-7B is weaker (36%) but SFT makes it worse, not better — same catastrophic forgetting pattern as our earlier Qwen3-14B SFT experiments
- Alex's parallel SFT experiments on Qwen3-8B (17 experiments, best +2.3%) are also not statistically significant (p=0.5)
- The "self-instruct ceiling effect" limits synthetic data quality — LLM-generated questions are systematically easier than real textbook problems

## Regex improvements made

The reward/evaluation regex was improved to handle all answer formats transparently:
- `\boxed{3.1%}` → 0.031 (percentage conversion)
- `\boxed{2.52 \times 10^{5}}` → 252000 (LaTeX scientific notation)
- `\boxed{14/33}` → 0.4242 (fractions)
- `\boxed{23,717}` → 23717 (thousands separators)
- `\boxed{0,001}` → 0.001 (european decimal comma)

## Files reference

| File | Content | Used by |
|------|---------|---------|
| `dataset_grpo_combined.json` | 57 questions filtered by answer diversity | GRPO |
| `dataset_dpo_pairs.json` | 233 preference pairs from base model generations | DPO |
| `dataset_sft_combined.json` | 281 questions (190 master + 65 hard + 26 extra) | SFT Qwen2.5 |
| `master_dataset_cleaned_numeric.json` | 190 single-answer numeric questions | source dataset |
| `hard_numeric_generated.jsonl` | 65 hard synthetic questions (from Alex) | source dataset |
