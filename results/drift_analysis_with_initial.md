# Drift analysis: where do "improved" questions come from?

## Setup

- **57 questions** in the GRPO training set
- **Initial state** (step 1 of training, base model) reconstructed from `training_samples_v3_hard.json`: each question has 4 generations with their predicted values, recomputed for correctness against `dataset_grpo_combined.json` targets
- **Final state** (after DPO training) from `Run 2/eval_finetuned_A_training.json`: same 4 generations per question on the trained model

## Initial distribution (before training)

| Initial n_correct | Count | Status |
|-------------------|-------|--------|
| 0/4 | 8 | base never gets it right |
| 1/4 | 6 | mixed (rare correct) |
| 2/4 | 14 | mixed |
| 3/4 | 13 | mixed (mostly correct) |
| 4/4 | 16 | base always gets it right |

→ 33/57 are mixed at start. 24/57 are saturated (8 always wrong + 16 always right).

## Transition matrix (initial → final after training)

| Initial \\ Final | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 | Total |
|------------------|-----|-----|-----|-----|-----|-------|
| 0/4 | 2 | 4 | 2 | 0 | **0** | 8 |
| 1/4 | 3 | 2 | 1 | 0 | **0** | 6 |
| 2/4 | 2 | 4 | 3 | 4 | **1** | 14 |
| 3/4 | 0 | 1 | 3 | 4 | **5** | 13 |
| 4/4 | 0 | 0 | 1 | 5 | **10** | 16 |

## Per-bucket evolution

For each initial bucket, where did the questions end up after training?

| Initial | n | Improved ↑ | Stable = | Degraded ↓ | Net |
|---------|---|------------|----------|------------|-----|
| 0/4 | 8 | 6 (→1/4: 4, →2/4: 2) | 2 | 0 | **+6** |
| 1/4 | 6 | 1 (→2/4: 1) | 2 | 3 (→0/4: 3) | **−2** |
| 2/4 | 14 | 5 (→3/4: 4, →4/4: 1) | 3 | 6 (→1/4: 4, →0/4: 2) | **−1** |
| 3/4 | 13 | 5 (→4/4: 5) | 4 | 4 (→2/4: 3, →1/4: 1) | **+1** |
| 4/4 | 16 | - | 10 | 6 (→3/4: 5, →2/4: 1) | **−6** |
| **Total** | **57** | **17** | **21** | **19** | **−2** |

### Key reading

- **The 16 "drifted to 4/4" don't represent 16 new wins.** They are 10 questions that were already at 4/4 (consolidated), 5 that were already at 3/4 (one extra correct generation), and **only 1 genuine improvement** (a 2/4 question that became 4/4).
- **The 7 "drifted to 0/4" cost the model real competence.** 5 of them came from questions where the base had partial knowledge (1/4 or 2/4) and the training erased it.
- **The 4/4 bucket leaks**: 6 out of 16 already-perfect questions lost at least one correct generation after training.
- **The hard end (0/4) does see small motion upward** (+6 improvements), but none reaches a confident state - they hover at 1/4 or 2/4.

## Net interpretation

Training:
- **Consolidates what was already mostly known** (3/4 → 4/4 transitions: 5 questions)
- **Erodes partial competence** (1/4 and 2/4 → 0/4: 5 questions; 4/4 → lower: 6 questions)
- **Produces only 1 unambiguous gain** (2/4 → 4/4)

The model becomes slightly more confident on what it already knew, and loses some of its hesitant correct answers on harder questions. This matches the "shortcut on easy / catastrophe on hard" pattern from the main analysis.

## Conclusion for the meeting

When asked *"isn't 16 questions becoming 4/4 a good sign?"*, the per-bucket table answers it directly: **15 of those 16 were already at 3/4 or 4/4 on the base** - the training only added at most one correct generation. Only **1 question** out of 57 represents a clear new capability. On the other side, **6 questions lost partial competence** (dropped from 1/4 or 2/4 down to 0/4, or from 4/4 down to 2/4).

The headline "16 → 4/4 vs 7 → 0/4" is misleading without the per-bucket decomposition. Net movement is essentially flat (+17 improved, −19 degraded, 21 stable), and the qualitative gain is one question.
