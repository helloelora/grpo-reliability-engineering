"""Evaluate base, SFT, or GRPO model on GSM8K benchmark.

Tests whether fine-tuning preserves or improves general mathematical reasoning.
Adapted from Alex's evaluate_gsm8k.py for standalone use on the cluster.

Usage (via env vars):
  LORA_PATH=""                    → base model
  LORA_PATH="/path/to/fold_4"    → SFT
  LORA_PATH="/path/to/ckpt-80"   → GRPO

GSM8K test set: 1319 questions. Each answer is a single integer.
"""

import os
import gc
import json
import re
import sys
import time
import torch

# ─── Patches (same as other eval scripts) ────────────────────────────────────

_orig_torch_mul = torch.Tensor.__mul__

def _safe_tensor_mul(self, other):
    if (
        isinstance(other, torch.Tensor)
        and self.dim() == 2
        and other.dim() == 2
        and self.shape[0] == other.shape[0]
        and self.shape[1] != other.shape[1]
        and abs(self.shape[1] - other.shape[1]) < 200
    ):
        _s = min(self.shape[1], other.shape[1])
        return _orig_torch_mul(self[:, :_s], other[:, :_s])
    return _orig_torch_mul(self, other)

torch.Tensor.__mul__ = _safe_tensor_mul

from unsloth import FastLanguageModel
from datasets import load_dataset as hf_load_dataset

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 2048
SEED = 3407

LORA_PATH = os.environ.get("LORA_PATH", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
RUN_NAME = os.environ.get("RUN_NAME", "gsm8k_eval")

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

SYSTEM_PROMPT = """/no_think
You are a helpful math tutor. Solve the following problem step-by-step.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response - your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
Always put your single final numerical answer inside \\boxed{}."""


# ─── Answer extraction ───────────────────────────────────────────────────────

def extract_gsm8k_answer(text):
    """Extract the numeric answer from GSM8K ground truth (#### <number>)."""
    match = re.search(r'####\s*([\-\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    return text.strip()


def extract_model_answer(response):
    """Extract final answer from model response (boxed or fallback)."""
    # Try \boxed{} first
    boxed = re.findall(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", response, re.DOTALL)
    if boxed:
        content = boxed[-1]  # last boxed
        nums = re.findall(r"-?[\d,]+\.?\d*", content)
        if nums:
            return nums[0].replace(",", "").strip()

    # Fallback patterns
    patterns = [
        r"[Ff]inal\s+[Aa]nswer\s*:\s*\$?\\?boxed\{?([\-\d,\.]+)\}?\$?",
        r"[Ff]inal\s+[Aa]nswer\s*:\s*([\-\d,\.]+)",
        r"[Tt]he\s+answer\s+is\s*:?\s*([\-\d,\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).replace(",", "").strip()

    # Last number in response
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def compare_gsm8k(predicted, ground_truth):
    """Compare predicted answer to GSM8K ground truth (exact integer match)."""
    try:
        pred = int(round(float(predicted.replace(",", "").strip())))
        truth = int(float(ground_truth.replace(",", "").strip()))
        return pred == truth
    except (ValueError, OverflowError):
        return False


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lora_provided = LORA_PATH and LORA_PATH.lower() not in ("none", "")
    if lora_provided and not os.path.exists(LORA_PATH):
        print(f"ERROR: LORA_PATH set but not found: {LORA_PATH}")
        sys.exit(1)

    # Load model
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_think(conversation, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig_apply(conversation, **kwargs)
    tokenizer.apply_chat_template = _apply_no_think

    if lora_provided:
        model.load_adapter(LORA_PATH)
        print(f"Loaded LoRA from {LORA_PATH}", flush=True)
        eval_label = "FINETUNED"
    else:
        print("Evaluating BASE model (no LoRA)", flush=True)
        eval_label = "BASE"
    FastLanguageModel.for_inference(model)

    # Load GSM8K
    print("Loading GSM8K test set...", flush=True)
    gsm8k = hf_load_dataset("openai/gsm8k", "main", split="test")
    print(f"GSM8K test questions: {len(gsm8k)}", flush=True)

    results = []
    correct_count = 0
    start_time = time.time()

    for q_idx, item in enumerate(gsm8k):
        question = item["question"]
        gt_answer = extract_gsm8k_answer(item["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        torch.manual_seed(SEED + q_idx)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy for benchmark
                use_cache=True,
            )
        new_tokens = output[0][input_ids["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        predicted = extract_model_answer(response)
        is_correct = compare_gsm8k(predicted, gt_answer)
        if is_correct:
            correct_count += 1

        results.append({
            "idx": q_idx,
            "question": question[:120],
            "ground_truth": gt_answer,
            "extracted_answer": predicted,
            "correct": is_correct,
        })

        if (q_idx + 1) % 50 == 0 or q_idx < 5:
            running_acc = correct_count / (q_idx + 1) * 100
            status = "OK" if is_correct else "XX"
            print(f"  [{q_idx+1}/{len(gsm8k)}] {status} | GT={gt_answer} Pred={predicted} | Running: {running_acc:.1f}%", flush=True)

    elapsed = time.time() - start_time
    accuracy = correct_count / len(results) * 100

    print(f"\n{'='*60}", flush=True)
    print(f"GSM8K {eval_label}: {correct_count}/{len(results)} = {accuracy:.1f}%", flush=True)
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/len(results):.1f}s/question)", flush=True)
    print(f"{'='*60}", flush=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"gsm8k_{RUN_NAME}.json")
    summary = {
        "benchmark": "gsm8k",
        "run_name": RUN_NAME,
        "lora_path": LORA_PATH,
        "total": len(results),
        "correct": correct_count,
        "accuracy": round(accuracy, 2),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}", flush=True)
