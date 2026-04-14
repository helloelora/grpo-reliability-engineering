"""
Evaluate the base Qwen2.5-7B model on the full dataset (281 questions).
Uses 5-fold splits (same seed as training) so results are directly comparable.
Includes McNemar test comparison with fine-tuned results if available.
"""
import os
import gc
import json
import re
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import KFold
from scipy.stats import binomtest

# PyTorch patch - must be before unsloth import
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
print("Applied torch.Tensor.__mul__ patch", flush=True)

from unsloth import FastLanguageModel


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_DIR, "results"))

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(PROJECT_DIR, "data", "dataset_sft_combined.json")
)

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)


# Configuration
SEED = 3407
MODEL_NAME     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN    = 8192
MAX_NEW_TOKENS = 8192
N_FOLDS        = 5

SYSTEM_PROMPT = """You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Always put your single final numerical answer inside \\boxed{}."""


# Regex extraction
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)

_LATEX_SCI_RE = re.compile(
    r"(-?[\d]+\.?\d*)\s*"
    r"(?:\\times|\\cdot|\u00d7|\*)\s*"
    r"10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?",
    re.DOTALL
)

_FRACTION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def extract_boxed_values(text, last_only=False):
    values = []
    for content in _BOXED_RE.findall(text):
        sci_match = _LATEX_SCI_RE.search(content)
        if sci_match:
            try:
                values.append(float(sci_match.group(1)) * (10 ** int(sci_match.group(2))))
            except (OverflowError, ValueError):
                pass
            continue

        is_percentage = False
        cleaned = content
        if re.search(r'[\d]\s*\\?%', content):
            is_percentage = True
            cleaned = re.sub(r'\\?%', '', content)

        normalized = re.sub(r'(\d{2,}),(\d{3})(?!\d)', r'\1\2', cleaned)
        normalized = re.sub(r'([1-9]),(\d{3})(?!\d)', r'\1\2', normalized)
        normalized = normalized.replace(',', '.')

        frac_match = _FRACTION_RE.search(normalized)
        if frac_match:
            num, den = float(frac_match.group(1)), float(frac_match.group(2))
            if den != 0:
                val = num / den
                if is_percentage: val /= 100
                values.append(val)
                continue

        for n in re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", normalized):
            try:
                val = float(n)
                if is_percentage: val /= 100
                values.append(val)
            except ValueError:
                pass

    if last_only and values:
        return [values[-1]]
    return values


def is_correct(pred, target, tol=0.05):
    if pred is None or target is None:
        return False
    if target == 0:
        return abs(pred) < 1e-9
    return abs(pred - target) / (abs(target) + 1e-9) <= tol


if __name__ == "__main__":
    if DATASET_PATH.endswith(".jsonl"):
        raw_data = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
    else:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    print(f"Dataset: {len(raw_data)} questions", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    print("Base Qwen2.5-7B loaded", flush=True)

    # Evaluate using the same fold splits as training (for per-fold comparison)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(raw_data)):
        test_items = [raw_data[i] for i in test_idx]
        print(f"\nFold {fold_i + 1}/{N_FOLDS} ({len(test_items)} test questions)", flush=True)

        fold_results = []
        for qi, item in enumerate(test_items):
            if "target_numeric" in item and item["target_numeric"]:
                target = item["target_numeric"][0]
            else:
                try:
                    target = float(item.get("answer", ""))
                except (ValueError, TypeError):
                    nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(item.get("answer", "")))
                    target = float(nums[0]) if nums else None
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

            torch.manual_seed(SEED + qi)
            with torch.no_grad():
                output = model.generate(
                    **input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    use_cache=True,
                )
            new_tokens = output[0][input_ids["input_ids"].shape[1]:]
            n_generated = len(new_tokens)
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            preds = extract_boxed_values(response, last_only=True)
            last_pred = preds[0] if preds else None
            correct = is_correct(last_pred, target)
            truncated = n_generated >= MAX_NEW_TOKENS and last_pred is None

            if truncated:
                print(f"    WARNING: Q{qi+1} truncated at {n_generated} tokens, no boxed found", flush=True)

            fold_results.append({
                "question": item["question"],
                "target": target,
                "answer": item["answer"],
                "correct": correct,
                "extracted_value": last_pred,
                "truncated": truncated,
                "response": response,
                "n_generated_tokens": n_generated,
            })

            status = "CORRECT" if correct else ("TRUNCATED" if truncated else "WRONG")
            print(f"    [{qi+1}/{len(test_items)}] {status} | target={target} pred={last_pred}", flush=True)

        n_correct = sum(1 for r in fold_results if r["correct"])
        n_truncated = sum(1 for r in fold_results if r["truncated"])
        fold_acc = n_correct / len(fold_results) * 100

        all_fold_results.append({
            "fold": fold_i,
            "n_test": len(test_items),
            "n_correct": n_correct,
            "n_truncated": n_truncated,
            "accuracy_pct": round(fold_acc, 1),
            "results": fold_results,
        })

        print(f"  Fold {fold_i + 1}: {n_correct}/{len(fold_results)} correct ({fold_acc:.1f}%), {n_truncated} truncated", flush=True)

        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, "sft_qwen25_base_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_fold_results, f, indent=2, ensure_ascii=False)

    # Summary
    fold_accs = [fr["accuracy_pct"] for fr in all_fold_results]
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"\nBASE QWEN2.5-7B RESULTS", flush=True)
    print(f"Per-fold: {fold_accs}", flush=True)
    print(f"Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%", flush=True)

    # McNemar test if finetuned results exist
    ft_path = os.path.join(OUTPUT_DIR, "sft_qwen25_finetuned_results.json")
    if os.path.exists(ft_path):
        print("\nRunning McNemar test against fine-tuned results...", flush=True)
        with open(ft_path, encoding="utf-8") as f:
            ft_data = json.load(f)

        b01, b10 = 0, 0
        for fold_i in range(min(len(all_fold_results), len(ft_data))):
            base_res = all_fold_results[fold_i]["results"]
            ft_res = ft_data[fold_i]["results"]
            for qi in range(min(len(base_res), len(ft_res))):
                bc = base_res[qi]["correct"]
                fc = ft_res[qi]["correct"]
                if bc and not fc: b10 += 1
                elif not bc and fc: b01 += 1

        if b10 + b01 > 0:
            result = binomtest(b01, b10 + b01, 0.5)
            print(f"  Base right, FT wrong: {b10}", flush=True)
            print(f"  Base wrong, FT right: {b01}", flush=True)
            print(f"  McNemar p-value: {result.pvalue:.4f}", flush=True)
            print(f"  Significant (p<0.05)? {'YES' if result.pvalue < 0.05 else 'No'}", flush=True)

    summary = {
        "model": MODEL_NAME,
        "method": "base (no fine-tuning)",
        "n_questions": len(raw_data),
        "n_folds": N_FOLDS,
        "fold_accuracies": fold_accs,
        "mean_accuracy": round(mean_acc, 1),
        "std_accuracy": round(std_acc, 1),
    }

    with open(os.path.join(OUTPUT_DIR, "sft_qwen25_base_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.", flush=True)
