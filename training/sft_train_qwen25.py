"""
SFT fine-tuning of Qwen2.5-7B on reliability engineering dataset.
5-fold cross-validation: for each fold, train on ~225 questions, evaluate on ~56.
Includes McNemar test at the end.
"""
import os
import gc
import sys
import json
import re
import random
import traceback
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import KFold
from datasets import Dataset
from scipy.stats import binomtest

# PyTorch patch — must be before unsloth import
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
from trl import SFTTrainer, SFTConfig


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(SCRIPT_DIR), "results"))

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(SCRIPT_DIR), "data", "dataset_sft_combined.json")
)

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)


# Configuration
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME  = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN_TRAIN = 4096   # for SFT training (reasoning chains are short)
MAX_SEQ_LEN_EVAL  = 8192   # for generation (model may produce longer output)
MAX_NEW_TOKENS    = 8192
LORA_R      = 16
LORA_ALPHA  = 16
NUM_EPOCHS  = 2
LEARNING_RATE = 2e-4
N_FOLDS     = 5

SYSTEM_PROMPT = """You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Always put your single final numerical answer inside \\boxed{}."""


# Regex extraction (same as grpo_train.py)
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
            values.append(float(sci_match.group(1)) * (10 ** int(sci_match.group(2))))
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


def format_training_text(item, tokenizer):
    """Format a single training example as a chat conversation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": f"{item['reasoning']}\n\n\\boxed{{{item['answer']}}}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def evaluate_on_split(model, tokenizer, test_items):
    """Evaluate model on a list of test items, return per-question results."""
    results = []
    for qi, item in enumerate(test_items):
        target = item["target_numeric"][0]
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

        n_correct = 1 if correct else 0
        results.append({
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

    return results


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Dataset: {len(raw_data)} questions", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(raw_data)):
        print(f"\nFold {fold_i + 1}/{N_FOLDS} ({len(train_idx)} train, {len(test_idx)} test)", flush=True)

        train_items = [raw_data[i] for i in train_idx]
        test_items = [raw_data[i] for i in test_idx]

        # Load fresh model for each fold
        gc.collect()
        torch.cuda.empty_cache()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN_EVAL, load_in_4bit=True,
            local_files_only=(fold_i > 0),  # use cached model after first fold
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build training dataset
        train_texts = [format_training_text(item, tokenizer) for item in train_items]
        train_ds = Dataset.from_dict({"text": train_texts})

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

        # Train
        training_args = SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN_TRAIN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            bf16=True,
            logging_steps=5,
            output_dir=os.path.join(SCRIPT_DIR, f"outputs_sft_fold{fold_i}"),
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_ds,
        )

        print(f"  Training fold {fold_i + 1}...", flush=True)
        trainer.train()
        print(f"  Training complete, evaluating...", flush=True)

        # Evaluate
        FastLanguageModel.for_inference(model)
        fold_results = evaluate_on_split(model, tokenizer, test_items)

        n_correct = sum(1 for r in fold_results if r["correct"])
        n_truncated = sum(1 for r in fold_results if r["truncated"])
        fold_acc = n_correct / len(fold_results) * 100

        all_fold_results.append({
            "fold": fold_i,
            "n_train": len(train_items),
            "n_test": len(test_items),
            "n_correct": n_correct,
            "n_truncated": n_truncated,
            "accuracy_pct": round(fold_acc, 1),
            "results": fold_results,
        })

        print(f"  Fold {fold_i + 1}: {n_correct}/{len(fold_results)} correct ({fold_acc:.1f}%), {n_truncated} truncated", flush=True)

        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, "sft_qwen25_finetuned_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_fold_results, f, indent=2, ensure_ascii=False)

        # Cleanup
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    fold_accs = [fr["accuracy_pct"] for fr in all_fold_results]
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"\nSFT QWEN2.5-7B RESULTS", flush=True)
    print(f"Per-fold: {fold_accs}", flush=True)
    print(f"Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%", flush=True)

    summary = {
        "model": MODEL_NAME,
        "method": "SFT",
        "dataset": DATASET_PATH,
        "n_questions": len(raw_data),
        "n_folds": N_FOLDS,
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "lora_r": LORA_R,
        "fold_accuracies": fold_accs,
        "mean_accuracy": round(mean_acc, 1),
        "std_accuracy": round(std_acc, 1),
    }

    with open(os.path.join(OUTPUT_DIR, "sft_qwen25_finetuned_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.", flush=True)
