## Datasets

| File | Questions | Description | Origin |
|------|-----------|-------------|--------|
| `dataset_grpo_combined.json` | 57 | Screened by answer diversity (mixed correct/wrong on 4 generations from the base model) | Filtered from `master_dataset_cleaned_numeric` and `hard_numeric_generated` |
| `dataset_sft_combined.json` | 281 | Master numeric (190) + hard generated (65) + 26 extras | Aggregated from earlier datasets |
| `master_dataset_v3.jsonl` | 501 | Alex's exp18 training set: 280 base + 221 paraphrases | Alex's repo |
| `master_dataset_v4.jsonl` | 866 | v3 + 365 new paraphrases | Alex's repo (latest) |
| `eval_holdout_v4_minus_v3_minus_281.json` | 365 | Held-out: questions in v4 but not in v3 - never seen by Alex's SFT or our previous GRPOs | Computed locally |

**Lineage:** v2 (280) ⊂ v3 (501) ⊂ v4 (866). Strict subsets, growing only via paraphrases.

## Common GRPO setup (all experiments)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Quantization | 4-bit |
| LoRA r/alpha | 32/32 (exp 5: 64/64) |
| Loss type | dr_grpo (Unsloth recommendation) |
| `mask_truncated_completions` | False |
| `importance_sampling_level` | sequence |
| `max_grad_norm` | 0.1 |
| `weight_decay` | 0.1 |
| `warmup_ratio` | 0.1 |
| `temperature` (training) | 1.0 |
| `top_p` | 0.95 |
| `num_generations` | 4 (exp 2: 8) |
| Batch size × grad accum | 1 × 4 |
| Eval | greedy seed-fixed sampling, temperature=1.0, max_new_tokens=4096 |

## Experiments

| Exp | Loss | Beta | LR | Steps | G | Max comp | LoRA r/α | Dataset | What I'm testing | Delta vs base |
|-----|------|------|-----|-------|---|----------|----------|---------|-----------------|---------------|
| **exp 1** | dr_grpo | 0.0 | 5e-6 | 220 | 4 | 3072 | 32/32 | 281q | Unsloth-recommended baseline (no KL constraint) | **−0.7%** |
| **exp 2** | dapo | 0.001 | 5e-6 | 110 | **8** | 3072 | 32/32 | 281q | More generations for richer gradient signal | killed (G=8 too slow) |
| **exp 3** | dr_grpo | 0.0 | **1e-5** | 220 | 4 | 3072 | 32/32 | 281q | Higher learning rate (2x baseline) | **+0.4%** |
| **exp 4** | dr_grpo | 0.0 | 5e-6 | 220 | 4 | **4096** | 32/32 | 281q | Larger context to avoid truncation | killed (too slow per step) |
| **exp 5** | dr_grpo | 0.0 | **1e-5** | 180 | 4 | 4096 | **64/64** | 281q | Higher LoRA capacity + aggressive lr | **+1.1%** |
| **exp 6** | dr_grpo | 0.0 | 5e-6 | 150 | 4 | 3072 | 32/32 | **501q v3** | DeepSeek-R1 approach: load Alex's SFT LoRA + GRPO on the same dataset | TBD (training in progress) |
| **exp 6bis** | dr_grpo | 0.0 | 5e-6 | 150 | 4 | 3072 | 32/32 | **365q held-out** | DeepSeek-R1 strict: SFT and RL on different datasets (mimics the paper) | TBD (training in progress) |

## Results

### In-distribution evaluation (model evaluated on its training dataset)

| Exp | Dataset | Base accuracy | FT accuracy | Delta | Improved | Degraded | McNemar p |
|-----|---------|--------------|-------------|-------|----------|----------|-----------|
| **exp 1** | 281q (sft_combined) | 63.7% (179/281) | 63.0% (177/281) | **−0.7%** | 16 | 18 | 0.86 |
| **exp 3** | 281q (sft_combined) | 63.7% (179/281) | 64.1% (180/281) | **+0.4%** | 22 | 21 | TBD |
| **exp 5** | 281q (sft_combined) | 63.7% (179/281) | 64.8% (182/281) | **+1.1%** | 22 | 19 | 0.76 |
| **exp 6**  | 501q (master_v3) | 56.1%  | 63.6% | TBD | TBD | TBD | TBD |
| **exp 6bis** | 365q (v4 holdout) | TBD | training in progress | TBD | TBD | TBD | TBD |


### Effect of the Unsloth-recommended fixes

All exp 1–6bis use the four Unsloth-recommended GRPO settings. What each one fixes - and what it does **not** fix:

| Setting | What it fixes | What it does NOT fix |
|---|---|---|
| `mask_truncated_completions=False` | Stops the trainer from discarding long truncated completions, which on hard questions are often the correct ones | Anything related to zero-variance steps |
| `importance_sampling_level="sequence"` (GSPO) | Numerical stability of importance sampling on long sequences | Anything related to zero-variance steps |
| `max_grad_norm=0.1` | Prevents gradient explosions on the rare useful steps | Slows down convergence |
| `loss_type="dr_grpo"` | Removes the length-normalization bias of vanilla GRPO | Anything related to zero-variance steps |


**−3.6% → +1.1%** - the fixes stopped GRPO from breaking the model

## Deep analysis

Three findings from the per-question JSON for exp 1 and exp 3 (file `eval_exp1_indist.json`).


**Finding 1 - GRPO improves easy questions and breaks hard ones.** Bucketing by base response length (proxy for difficulty), the same pattern appears in both exp 1 and exp 5:

| Difficulty | n | Base | Exp 1 FT | Exp 5 FT |
|------------|---|------|----------|----------|
| Easy (<1500 chars) | 126 | 83.3% | 86.5% (+3.2%) | 88.9% (**+5.6%**) |
| Medium (1500-3000) | 100 | 53.0% | 54.0% (+1.0%) | 57.0% (+4.0%) |
| Hard (>3000) | 55 | 38.2% | 25.5% (**−12.7%**) | 23.6% (**−14.5%**) |

**Exp 5 (LoRA r=64, lr=1e-5) reproduces and amplifies the pattern**: bigger gains on easy/medium (+5.6% / +4.0%), bigger losses on hard (−14.5%). Higher capacity makes both effects stronger but the global delta stays small (+1.1%) because they cancel out.

**Finding 2 - Errors are catastrophic, not marginal.** 

| Question snippet (unique in dataset) | Target | Base (correct) | FT (wrong) | Off by |
|---------------------------------------|--------|---------------|------------|--------|
| `A chemical plant has a safety system consisting of two subsystems in series. Subsystem 1 is a 2-out-of-3 redundant configuration (system works if at least 2 of` | 0.903 | 0.912 | 0.000951 | **1000×** |
| `A chemical plant has a safety system composed of two identical pressure relief v` | 0.868 | 0.888 | 0.000774 | **1000×** |
| `A component failure distribution is exponential with mean rate λ = 6.0%/K. For a` | 2 | 2 | 67 | 33× |
| `Using the Goal Seek function, find the necessary sample size n for an LTPD = 2.21% and a beta risk of 0.1 for the acceptance number c = 1` | 175 | 170 | 50 | 3.4× |
| `A telecommunications satellite contains a critical oscillator component whose lifetime follows a Weibull distribution with shape parameter β = 2.0 and characteristic life η = 50,000 hours` | 3 | 3 | 11 | 3.7× |

GRPO learned shortcuts on easy/medium questions and blindly applies them to hard questions where they don't work at all.

**Finding 3 - 42% of our training steps had zero gradient.** Logged every training step on the 57-question run (`training_samples_v3_hard.json`, 80 steps × 4 generations = 320 records) and counted how many of the 4 generations were correct at each step:

| n_correct/4 at step | # steps | Status |
|---|---|---|
| 0/4 | 11 | wasted (variance=0) |
| 1/4 | 9 | useful gradient |
| 2/4 | 18 | useful gradient |
| 3/4 | 19 | useful gradient |
| 4/4 | 23 | wasted (variance=0) |

→ **34/80 steps (42%) produced zero gradient** because all 4 generations agreed. This is measured directly on our run, not extrapolated. It is the empirical signature of the missing dynamic sampling.

**Note on missing DAPO features.** TRL/Unsloth does not natively support **dynamic sampling**, the technique introduced in the **DAPO paper** (*"DAPO: An Open-Source LLM Reinforcement Learning System at Scale"*, ByteDance/Tsinghua, March 2024). The paper directly criticizes vanilla GRPO on this point: *"many prompts produce zero advantage and waste training compute"*. Dynamic sampling fixes this by re-drawing a prompt whenever all G generations give the same reward (variance = 0), and only counting steps where the gradient is non-zero.



```
Without dynamic sampling (our current setup):
  Step 1: pick Q1 from dataset → generate 4 responses → all correct (4/4) → variance=0 → step wasted
  Step 2: pick Q2 → 1/4 correct → useful gradient → step counts
  Step 3: pick Q3 → all wrong (0/4) → variance=0 → step wasted
  ...

With dynamic sampling (DAPO):
  Step 1: pick Q1 → 4/4 → DROP, pick Q5 → 0/4 → DROP, pick Q12 → 2/4 → keep, do the step
  Step 2: pick Q2 → 1/4 → keep, do the step
  Step 3: pick Q3 → 0/4 → DROP, pick Q8 → 3/4 → keep, do the step
  ...
```