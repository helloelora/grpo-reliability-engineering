# GRPO experiments summary

## Common configuration (all experiments)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA r/alpha | 32/32 (exp 5: 64/64) |
| `mask_truncated_completions` | False |
| `temperature` (training) | 1.0 |
| `top_p` | 0.95 |
| `importance_sampling_level` | sequence |
| `max_grad_norm` | 0.1 |
| `weight_decay` | 0.1 |
| `warmup_ratio` | 0.1 |
| Eval temperature | 1.0 |
| Eval `top_p` | 0.95 |
| Eval `max_new_tokens` | 4096 |
| Eval seed | 3407 + question_index (fixed) |

## Hyperparameters per experiment

| Exp | Loss | Beta | LR | Steps | G | Max comp len | Dataset | LoRA r/α |
|-----|------|------|-----|-------|---|--------------|---------|----------|
| **1** | dr_grpo | 0.0 | 5e-6 | 220 | 4 | 3072 | 281q (`dataset_sft_combined`) | 32/32 |
| **3** | dr_grpo | 0.0 | 1e-5 | 220 | 4 | 3072 | 281q (`dataset_sft_combined`) | 32/32 |
| **5** | dr_grpo | 0.0 | 1e-5 | 180 | 4 | 4096 | 281q (`dataset_sft_combined`) | **64/64** |
| **6** | dr_grpo | 0.0 | 5e-6 | 150 (target) | 4 | 3072 | 501q (`master_dataset_v3`) | 32/32 (+ Alex SFT LoRA pre-loaded) |

## In-distribution evaluation results

Evaluated on the same dataset used for training. Same base model, same seeds.

| Exp | Dataset | Base accuracy | FT accuracy | Delta | Improved | Degraded | McNemar p |
|-----|---------|---------------|-------------|-------|----------|----------|-----------|
| **Exp 1** | 281q | 63.7% (179/281) | 63.0% (177/281) | **-0.7%** | 16 | 18 | 0.8642 |
| **Exp 3** | 281q | 63.7% (179/281) | 64.1% (180/281) | **+0.4%** | TBD | TBD | TBD |
| **Exp 5** | 281q | TBD | TBD (training in progress) | TBD | TBD | TBD | TBD |
| **Exp 6** | 501q | TBD (waiting Colab) | TBD (training in progress) | TBD | TBD | TBD | TBD |

**Note for exp 3:** Improved/degraded counts and McNemar p will be computed once we cross the ft_results with base results from exp 1 (same dataset, same seeds → directly comparable).

## Held-out evaluation (365 questions, never seen by anyone)

To be done after in-dist eval, only if model shows improvement.

| Exp | Held-out accuracy base | Held-out accuracy FT | Delta | McNemar p |
|-----|------------------------|---------------------|-------|-----------|
| Exp 1 | TBD (Colab) | TBD | TBD | TBD |
| Exp 3 | TBD (Colab) | TBD | TBD | TBD |
| Exp 5 | TBD (Colab) | TBD | TBD | TBD |
| Exp 6 | TBD (Colab) | TBD | TBD | TBD |

## Should we evaluate on held-out?

**Exp 1: NO.** Delta -0.7%, p=0.86. The model is statistically indistinguishable from base on its own training data. No reason to expect generalization to unseen data.

**Exp 3: PROBABLY NOT.** Delta +0.4% (1 question difference). Statistically meaningless on the training data itself. Held-out eval would just confirm no effect.

**Exp 5: WAIT.** Training still running. With LoRA r=64 (double capacity), might show a different pattern.

**Exp 6: WAIT.** Most promising experiment because it starts from Alex's SFT LoRA (already +6.2% on similar data). Even partial training could maintain Alex's gains.

## Conclusion so far

Exp 1 and 3 confirm what we observed in earlier runs: GRPO on this dataset size (~280 questions) does not produce measurable improvement, even with all the configuration fixes (mask_truncated=False, temperature=1.0, dr_grpo loss, importance_sampling=sequence). The variance is so small (1-2 questions) that any apparent change is noise.

The remaining hope is exp 6 (SFT cold start + GRPO) which leverages Alex's already-trained model.
