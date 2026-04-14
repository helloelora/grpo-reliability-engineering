"""
GRPO Training: Qwen3-14B for Reliability Engineering

Group Relative Policy Optimization (GRPO) — reinforcement learning approach
that rewards correct reasoning without overwriting the base model's knowledge.

Unlike SFT (which replaces the model's outputs with ours), GRPO:
  1. Generates G completions per question
  2. Scores each with reward functions (correctness + format)
  3. Reinforces completions that score above the group mean
  4. Keeps the model close to its original behavior (via KL penalty)

This avoids catastrophic forgetting — the key failure mode of SFT on small datasets.

Configuration:
    - Model: Qwen3-14B (4-bit quantized via Unsloth)
    - LoRA: r=32, alpha=32 (moderate capacity)
    - GRPO: G=4 completions, beta=0.001 (light KL), DAPO loss
    - Reward: LLM judge (Claude 3.5 Sonnet) + format adherence
    - Eval: greedy decoding, 5-fold CV, same judge
    - Data format: conversational (system + user prompt)

Reference:
    DeepSeek-R1 paper: https://arxiv.org/abs/2402.03300
"""

import os

# ─── RUCHE HPC Environment Setup ────────────────────────────────
if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")
    os.makedirs(hf_cache, exist_ok=True)
else:
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
# ─────────────────────────────────────────────────────────────────

import gc
import json
import random
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from sklearn.model_selection import KFold
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# Configuration
# ============================================================

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset_alex.json"),
)

# LoRA configuration
LORA_R = 32
LORA_ALPHA = 32

# GRPO configuration
NUM_GENERATIONS = 4          # G completions per prompt
MAX_COMPLETION_LENGTH = 4096  # max tokens per completion (fits 4×4096 in A100 40GB)
MAX_PROMPT_LENGTH = 1024      # max tokens for the prompt
GRPO_BETA = 0.001            # KL penalty (DeepSeek-R1 value)
GRPO_TEMPERATURE = 0.7       # sampling temperature for generations
GRPO_LOSS_TYPE = "grpo"      # "grpo" or "dapo"
MAX_STEPS = 100               # GRPO training steps per fold (increase if reward still climbing)
LEARNING_RATE = 5e-6

# Evaluation configuration — greedy (deterministic)
EVAL_THINKING_MODE = True
EVAL_MAX_NEW_TOKENS = 16384
EVAL_TEMPERATURE = 0.0

# Judge configuration (OpenRouter)
JUDGE_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "REDACTED",
)
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
    default_headers={
        "HTTP-Referer": "https://github.com/helloelora",
        "X-Title": "Reliability-GRPO",
    },
)

# System prompt — IDENTICAL to all previous experiments for fair comparison
SYSTEM_PROMPT = """You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Use LaTeX for mathematical formulas.
Be concise: focus on the key calculation steps, avoid repeating the question or adding unnecessary preamble.
Always conclude with a clearly stated final answer including numerical values and units when applicable."""


# ============================================================
# Dataset preparation for GRPO
# ============================================================


def load_and_prepare_dataset(path):
    """
    Load dataset and format for GRPOTrainer.

    GRPOTrainer expects a 'prompt' column in conversational format
    (list of message dicts) and any extra columns for reward functions.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ],
            "answer": item["answer"],
            "question": item["question"],
            "reasoning": item.get("reasoning", ""),
        })

    return Dataset.from_list(records)


# ============================================================
# Reward functions
# ============================================================


def reward_correctness(completions, answer, question, **kwargs):
    """
    Core reward: use LLM judge to check correctness.

    Returns +2.0 for correct, -1.0 for incorrect.
    This is the same Claude 3.5 Sonnet judge used in all SFT experiments.
    """
    rewards = []
    for completion, gt_answer, q in zip(completions, answer, question):
        response_text = completion[0]["content"] if isinstance(completion, list) else str(completion)

        # Extract the final answer from the response
        final_answer = _extract_final_answer(response_text)

        judge_prompt = f"""You are a STRICT impartial exam grader for Reliability Engineering.

Compare Student's Answer with the Ground Truth.

--- QUESTION ---
{q}

--- GROUND TRUTH ---
{gt_answer}

--- STUDENT'S ANSWER ---
{final_answer}

GRADING RULES:
1. GIBBERISH: If the student's answer is repetitive, nonsensical, empty, or contains unrelated spam, mark is_correct=false.
2. STRICT MATH: Numerical final result must be within 5.0% margin of the ground truth (when applicable).
3. LOGIC: If the student provides reasoning, it must be coherent; do not invent missing steps.
4. PARTIAL CREDIT: If the student's reasoning is sound and arrives at the right concept/formula but has minor rounding differences, still mark is_correct=true.
5. OUTPUT: Return ONLY valid JSON with keys:
   - "is_correct": boolean
   - "explanation": string (brief justification)
"""
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*?"is_correct".*?\}', raw_content, re.DOTALL)
                data = json.loads(match.group()) if match else {"is_correct": False}

            is_correct = bool(data.get("is_correct", False))
            rewards.append(2.0 if is_correct else -1.0)

        except Exception as e:
            print(f"  [Judge error] {e}")
            rewards.append(0.0)  # Neutral on API failure

    return rewards


def reward_format(completions, **kwargs):
    """
    Format reward: encourage structured responses with a final answer.

    +1.0 if response contains "Final Answer" marker
    +0.5 if response has some mathematical content (LaTeX)
    -0.5 if response is too short (< 50 chars) or empty
    """
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = 0.0

        # Check for final answer marker
        if re.search(r"\*\*Final Answer[:\*]*\*\*", response):
            score += 1.0
        elif re.search(r"final answer|the answer is", response, re.IGNORECASE):
            score += 0.5

        # Check for mathematical content (LaTeX)
        if re.search(r"\$.*?\$|\\frac|\\lambda|\\exp|\\int|\\sum", response):
            score += 0.5

        # Penalize empty or very short responses
        if len(response.strip()) < 50:
            score -= 0.5

        # Penalize excessive repetition
        lines = response.strip().split("\n")
        if len(lines) > 5:
            unique_lines = set(l.strip() for l in lines if l.strip())
            if len(unique_lines) < len(lines) * 0.3:
                score -= 1.0

        rewards.append(score)

    return rewards


def reward_reasoning_length(completions, **kwargs):
    """
    Mild reward for substantive reasoning (not too short, not too long).

    Encourages responses between 200-2000 chars (the sweet spot for
    reliability engineering problems).
    """
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        length = len(response.strip())

        if length < 100:
            rewards.append(-0.5)
        elif length < 200:
            rewards.append(0.0)
        elif length <= 2000:
            rewards.append(0.5)
        elif length <= 4000:
            rewards.append(0.0)
        else:
            rewards.append(-0.3)  # Mild penalty for verbosity

    return rewards


# ============================================================
# Answer extraction (identical to SFT experiments)
# ============================================================


def _extract_final_answer(text):
    """Extract the final answer from a Qwen3 response, handling <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()

    if not cleaned:
        after_think = re.split(r"</think>", text)
        if len(after_think) > 1:
            cleaned = after_think[-1].strip()
        else:
            fa_match = re.search(r"\*\*Final Answer[:\*]*\*\*(.+)", text, re.DOTALL)
            if fa_match:
                cleaned = "**Final Answer:**" + fa_match.group(1).strip()
            else:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
                cleaned = paragraphs[-1] if paragraphs else "[No final answer produced]"

    return _truncate_repetitions(cleaned)


def _truncate_repetitions(text, max_repeats=3):
    """Detect and truncate excessive line/word repetitions."""
    lines = text.split("\n")
    result_lines = []
    prev_line = None
    repeat_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue
        else:
            repeat_count = 0
            prev_line = stripped
        result_lines.append(line)

    result = "\n".join(result_lines)
    result = re.sub(r"(\b\w{3,30}\b)(\s+\1){4,}", r"\1", result)

    parts = result.split("**Final Answer:**")
    if len(parts) > 2:
        result = parts[0] + "**Final Answer:**" + parts[1]

    return result.strip()


# ============================================================
# Generation (for post-training evaluation)
# ============================================================


@torch.inference_mode()
def generate_answer(model, tokenizer, question):
    """Generate an answer using greedy decoding (deterministic)."""
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=EVAL_THINKING_MODE, return_tensors="pt",
    ).to("cuda")

    gen_kwargs = dict(
        max_new_tokens=EVAL_MAX_NEW_TOKENS,
        use_cache=True,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=10,
    )

    outputs = model.generate(input_ids, **gen_kwargs)
    output_ids = outputs[0][input_ids.shape[1]:].tolist()
    raw = tokenizer.decode(output_ids, skip_special_tokens=False)

    n_tokens = len(output_ids)
    had_thinking = "<think>" in raw
    raw_clean = tokenizer.decode(output_ids, skip_special_tokens=True)
    answer = _extract_final_answer(raw_clean)

    return answer, n_tokens, had_thinking, raw_clean


# ============================================================
# LLM Judge (for post-training evaluation — identical to SFT)
# ============================================================


def judge_single(sample, student_answer):
    """Evaluate a student answer against ground truth using an LLM judge."""
    judge_prompt = f"""You are a STRICT impartial exam grader for Reliability Engineering.

Compare Student's Answer with the Ground Truth.

--- QUESTION ---
{sample['question']}

--- GROUND TRUTH ---
{sample['answer']}

--- STUDENT'S ANSWER ---
{student_answer}

GRADING RULES:
1. GIBBERISH: If the student's answer is repetitive, nonsensical, empty, or contains unrelated spam, mark is_correct=false.
2. STRICT MATH: Numerical final result must be within 5.0% margin of the ground truth (when applicable).
3. LOGIC: If the student provides reasoning, it must be coherent; do not invent missing steps.
4. PARTIAL CREDIT: If the student's reasoning is sound and arrives at the right concept/formula but has minor rounding differences, still mark is_correct=true.
5. OUTPUT: Return ONLY valid JSON with keys:
   - "is_correct": boolean
   - "explanation": string (brief justification)
"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*?"is_correct".*?\}', raw_content, re.DOTALL)
                data = json.loads(match.group()) if match else {"is_correct": False, "explanation": "Parse error"}

            return {
                "question": sample["question"],
                "target": sample["answer"],
                "student_answer": student_answer,
                "is_correct": bool(data.get("is_correct", False)),
                "explanation": str(data.get("explanation", "")),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "question": sample.get("question", ""),
                "target": sample.get("answer", ""),
                "student_answer": student_answer,
                "is_correct": False,
                "explanation": f"Judge API error: {e}",
            }


# ============================================================
# Memory management
# ============================================================


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================
# Main: 5-fold cross-validation with GRPO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO TRAINING — Qwen3-14B Reliability Engineering")
    print(f"  LoRA r={LORA_R}, α={LORA_ALPHA}")
    print(f"  GRPO: G={NUM_GENERATIONS}, β={GRPO_BETA}, T={GRPO_TEMPERATURE}")
    print(f"  Loss: {GRPO_LOSS_TYPE}, Steps: {MAX_STEPS}")
    print(f"  SEED={SEED}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    full_dataset = load_and_prepare_dataset(DATASET_PATH)
    print(f"Loaded {len(full_dataset)} samples")

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accuracies = []
    all_results = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_grpo_{run_timestamp}.json"

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{N_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'=' * 60}")

        clear_cuda()

        # ── 1. Load fresh model ──────────────────────────────────
        print("\n  Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
        )

        # ── 2. Add LoRA ─────────────────────────────────────────
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        # ── 3. Prepare training data ────────────────────────────
        train_ds = full_dataset.select(train_idx)
        print(f"  Training on {len(train_ds)} samples with GRPO")

        # ── 4. GRPO Training ────────────────────────────────────
        training_args = GRPOConfig(
            # Generation
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            temperature=GRPO_TEMPERATURE,

            # GRPO specific
            beta=GRPO_BETA,
            loss_type=GRPO_LOSS_TYPE,
            num_iterations=1,

            # Training
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=NUM_GENERATIONS,
            gradient_accumulation_steps=4,
            max_steps=MAX_STEPS,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,

            # Infrastructure
            bf16=True,
            seed=SEED,
            logging_steps=5,
            output_dir=f"outputs_grpo_fold_{fold}",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,  # Keep extra columns for reward funcs
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_correctness,
                reward_format,
                reward_reasoning_length,
            ],
            args=training_args,
            train_dataset=train_ds,
        )

        print("  Starting GRPO training...")
        trainer.train()
        print("  GRPO training complete.")

        # Save LoRA adapter for this fold
        lora_path = f"grpo_saved_lora_fold_{fold}"
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        print(f"  LoRA saved to {lora_path}")

        # ── 5. Evaluate on validation set ────────────────────────
        print(f"\n  Evaluating fold {fold + 1} ({len(val_idx)} samples)...")

        fold_correct = 0
        fold_results = []
        token_counts = []

        for i in tqdm(range(len(val_idx)), desc=f"Eval fold {fold + 1}"):
            idx = int(val_idx[i])
            sample = {
                "question": full_dataset[idx]["question"],
                "answer": full_dataset[idx]["answer"],
            }

            ans, n_tok, had_thinking, raw_response = generate_answer(
                model, tokenizer, sample["question"]
            )
            judged = judge_single(sample, ans)
            judged.update({
                "fold": fold + 1,
                "sample_index": idx,
                "token_count": n_tok,
                "had_thinking": had_thinking,
                "raw_response": raw_response,
            })
            fold_results.append(judged)
            token_counts.append(n_tok)
            if judged["is_correct"]:
                fold_correct += 1

        fold_acc = fold_correct / len(val_idx) * 100
        fold_accuracies.append(fold_acc)
        all_results.extend(fold_results)

        print(f"  Fold {fold + 1}: {fold_acc:.2f}% ({fold_correct}/{len(val_idx)}), "
              f"tokens: mean={np.mean(token_counts):.0f}, max={max(token_counts)}")

        # ── 6. Save incremental results ──────────────────────────
        json_results = {
            "metadata": {
                "variant": "grpo",
                "description": "GRPO (Group Relative Policy Optimization) — RL-based training",
                "model": MODEL_NAME,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "grpo_config": {
                    "num_generations": NUM_GENERATIONS,
                    "beta": GRPO_BETA,
                    "temperature": GRPO_TEMPERATURE,
                    "loss_type": GRPO_LOSS_TYPE,
                    "max_steps": MAX_STEPS,
                    "max_completion_length": MAX_COMPLETION_LENGTH,
                },
                "learning_rate": LEARNING_RATE,
                "eval_temperature": EVAL_TEMPERATURE,
                "judge_model": JUDGE_MODEL,
                "reward_functions": [
                    "reward_correctness (LLM judge, +2/-1)",
                    "reward_format (final answer marker, +1/-0.5)",
                    "reward_reasoning_length (200-2000 chars, +0.5/-0.5)",
                ],
                "n_folds": N_FOLDS,
                "folds_completed": fold + 1,
                "total_samples": len(full_dataset),
                "seed": SEED,
                "timestamp": run_timestamp,
            },
            "summary": {
                "fold_accuracies": fold_accuracies,
                "mean_accuracy": float(np.mean(fold_accuracies)),
                "std_accuracy": float(np.std(fold_accuracies)) if len(fold_accuracies) > 1 else 0.0,
            },
            "all_results": all_results,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        # Cleanup
        del trainer, model, tokenizer
        clear_cuda()

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"GRPO COMPLETE — Mean: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Fold accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 60}")
