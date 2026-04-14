"""
SFT fine-tuning of Qwen3-8B on reliability engineering dataset.
5-fold cross-validation. Reproduces Alex's best config (exp 12).
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
from scipy.stats import binomtest

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
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(SCRIPT_DIR), "results"))
DATASET_PATH = os.environ.get("DATASET_PATH")
RUN_NAME = os.environ.get("RUN_NAME", "sft_qwen3")

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME     = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN    = 4096
MAX_NEW_TOKENS = 8192
LORA_R         = 16
LORA_ALPHA     = 32
NUM_EPOCHS     = 4
LEARNING_RATE  = 2e-4
NEFTUNE_ALPHA  = 5
N_FOLDS        = 5

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Always put your single final numerical answer inside \\boxed{}."""

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_LATEX_SCI_RE = re.compile(
    r"(-?[\d]+\.?\d*)\s*(?:\\times|\\cdot|\u00d7|\*)\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?",
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": f"{item['reasoning']}\n\n\\boxed{{{item['answer']}}}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def evaluate_fold(model, tokenizer, test_items):
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
                do_sample=False,
                use_cache=True,
            )
        new_tokens = output[0][input_ids["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds = extract_boxed_values(response, last_only=True)
        last_pred = preds[0] if preds else None
        correct = is_correct(last_pred, target)
        truncated = len(new_tokens) >= MAX_NEW_TOKENS and last_pred is None

        results.append({
            "question": item["question"][:100],
            "target": target,
            "extracted_value": last_pred,
            "correct": correct,
            "truncated": truncated,
            "response": response,
        })
        status = "CORRECT" if correct else ("TRUNCATED" if truncated else "WRONG")
        print(f"    [{qi+1}/{len(test_items)}] {status} | target={target} pred={last_pred}", flush=True)
    return results


if __name__ == "__main__":
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"ERROR: DATASET_PATH not set or not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Dataset: {len(raw_data)} questions, RUN_NAME={RUN_NAME}", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_base_results = []
    all_ft_results = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(raw_data)):
        train_items = [raw_data[i] for i in train_idx]
        test_items = [raw_data[i] for i in test_idx]
        print(f"\nFold {fold_i+1}/{N_FOLDS} ({len(train_items)} train, {len(test_items)} test)", flush=True)

        gc.collect(); torch.cuda.empty_cache()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
            local_files_only=(fold_i > 0),
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _orig_apply = tokenizer.apply_chat_template
        def _apply_no_think(conversation, **kwargs):
            kwargs["enable_thinking"] = False
            return _orig_apply(conversation, **kwargs)
        tokenizer.apply_chat_template = _apply_no_think

        # Evaluate base model on test fold
        FastLanguageModel.for_inference(model)
        print(f"  Evaluating base...", flush=True)
        base_results = evaluate_fold(model, tokenizer, test_items)
        base_correct = sum(1 for r in base_results if r["correct"])
        print(f"  Base: {base_correct}/{len(test_items)}", flush=True)

        # Train SFT
        del model; gc.collect(); torch.cuda.empty_cache()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.apply_chat_template = _apply_no_think

        train_texts = [format_training_text(item, tokenizer) for item in train_items]
        train_ds = Dataset.from_dict({"text": train_texts})

        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

        training_args = SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            neftune_noise_alpha=NEFTUNE_ALPHA,
            seed=SEED,
            bf16=True,
            logging_steps=10,
            output_dir=os.path.join(SCRIPT_DIR, f"outputs_{RUN_NAME}_fold{fold_i}"),
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            args=training_args, train_dataset=train_ds,
        )
        print(f"  Training...", flush=True)
        trainer.train()

        # Evaluate fine-tuned on test fold
        FastLanguageModel.for_inference(model)
        print(f"  Evaluating fine-tuned...", flush=True)
        ft_results = evaluate_fold(model, tokenizer, test_items)
        ft_correct = sum(1 for r in ft_results if r["correct"])
        print(f"  Fine-tuned: {ft_correct}/{len(test_items)}", flush=True)

        all_base_results.append({"fold": fold_i, "n_test": len(test_items),
            "n_correct": base_correct, "accuracy_pct": round(base_correct/len(test_items)*100, 1),
            "results": base_results})
        all_ft_results.append({"fold": fold_i, "n_test": len(test_items),
            "n_correct": ft_correct, "accuracy_pct": round(ft_correct/len(test_items)*100, 1),
            "results": ft_results})

        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, f"{RUN_NAME}_results.json"), "w", encoding="utf-8") as f:
            json.dump({"base": all_base_results, "ft": all_ft_results}, f, indent=2, ensure_ascii=False)

        del model, tokenizer, trainer; gc.collect(); torch.cuda.empty_cache()

    # Summary
    base_accs = [r["accuracy_pct"] for r in all_base_results]
    ft_accs = [r["accuracy_pct"] for r in all_ft_results]
    print(f"\nBase per-fold: {base_accs}, mean={np.mean(base_accs):.1f}%", flush=True)
    print(f"FT per-fold:   {ft_accs}, mean={np.mean(ft_accs):.1f}%", flush=True)
    print(f"Delta: {np.mean(ft_accs)-np.mean(base_accs):+.1f}%", flush=True)

    # McNemar
    b01, b10 = 0, 0
    for fi in range(len(all_base_results)):
        for qi in range(len(all_base_results[fi]["results"])):
            bc = all_base_results[fi]["results"][qi]["correct"]
            fc = all_ft_results[fi]["results"][qi]["correct"]
            if bc and not fc: b10 += 1
            elif not bc and fc: b01 += 1
    if b10 + b01 > 0:
        result = binomtest(b01, b10 + b01, 0.5)
        print(f"McNemar: base_right_ft_wrong={b10}, base_wrong_ft_right={b01}, p={result.pvalue:.4f}", flush=True)

    with open(os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_name": RUN_NAME, "model": MODEL_NAME, "dataset": DATASET_PATH,
            "n_questions": len(raw_data), "n_folds": N_FOLDS, "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE, "lora_r": LORA_R, "neftune": NEFTUNE_ALPHA,
            "base_accs": base_accs, "ft_accs": ft_accs,
            "base_mean": round(np.mean(base_accs), 1), "ft_mean": round(np.mean(ft_accs), 1),
            "delta": round(np.mean(ft_accs) - np.mean(base_accs), 1),
            "mcnemar_b01": b01, "mcnemar_b10": b10,
            "mcnemar_p": round(result.pvalue, 4) if b10 + b01 > 0 else None,
        }, f, indent=2)
    print("Done.", flush=True)
