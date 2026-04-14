"""
Evaluate a single LoRA model on a single dataset.
Generic script - takes LORA_PATH and DATASET_PATH from env vars.
Also evaluates the base model for comparison and runs McNemar test.
"""
import os
import gc
import json
import re
import numpy as np
import torch
from datetime import datetime
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

LORA_PATH = os.environ.get("LORA_PATH")
DATASET_PATH = os.environ.get("DATASET_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_DIR, "results"))
RUN_NAME = os.environ.get("RUN_NAME", "eval")

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 4096
SEED = 3407

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response - your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
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


def evaluate(model, tokenizer, questions, label):
    results = []
    for qi, q in enumerate(questions):
        target = q["target_numeric"][0] if "target_numeric" in q and q["target_numeric"] else None
        if target is None:
            try: target = float(q.get("answer", ""))
            except: pass

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["question"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        torch.manual_seed(SEED + qi)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                use_cache=True,
            )
        new_tokens = output[0][input_ids["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        preds = extract_boxed_values(response, last_only=True)
        last_pred = preds[0] if preds else None
        correct = is_correct(last_pred, target)
        truncated = len(new_tokens) >= MAX_NEW_TOKENS and last_pred is None

        results.append({
            "question": q["question"][:100],
            "target": target,
            "extracted_value": last_pred,
            "correct": correct,
            "truncated": truncated,
            "response": response,
        })

        status = "CORRECT" if correct else ("TRUNCATED" if truncated else "WRONG")
        print(f"  [{qi+1}/{len(questions)}] {status} | target={target} pred={last_pred}", flush=True)

    n_correct = sum(1 for r in results if r["correct"])
    n_trunc = sum(1 for r in results if r["truncated"])
    acc = n_correct / len(results) * 100
    print(f"\n{label}: {n_correct}/{len(results)} correct ({acc:.1f}%), {n_trunc} truncated", flush=True)
    return results, acc


if __name__ == "__main__":
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"ERROR: DATASET_PATH not set or not found: {DATASET_PATH}")
        import sys; sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Dataset: {len(questions)} questions", flush=True)

    # Evaluate base model
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

    FastLanguageModel.for_inference(model)
    print("\nEvaluating BASE model...", flush=True)
    base_results, base_acc = evaluate(model, tokenizer, questions, "BASE")

    # Evaluate fine-tuned model
    if LORA_PATH and os.path.exists(LORA_PATH):
        del model; gc.collect(); torch.cuda.empty_cache()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.apply_chat_template = _apply_no_think
        model.load_adapter(LORA_PATH)
        FastLanguageModel.for_inference(model)
        print(f"\nEvaluating FINETUNED model ({LORA_PATH})...", flush=True)
        ft_results, ft_acc = evaluate(model, tokenizer, questions, "FINETUNED")

        # McNemar test
        b01, b10 = 0, 0
        for i in range(len(base_results)):
            bc = base_results[i]["correct"]
            fc = ft_results[i]["correct"]
            if bc and not fc: b10 += 1
            elif not bc and fc: b01 += 1

        print(f"\nMcNemar: base_right_ft_wrong={b10}, base_wrong_ft_right={b01}", flush=True)
        if b10 + b01 > 0:
            result = binomtest(b01, b10 + b01, 0.5)
            print(f"p-value: {result.pvalue:.4f}", flush=True)
            delta = ft_acc - base_acc
            print(f"Delta: {delta:+.1f}%", flush=True)

        # Save
        output = {
            "run_name": RUN_NAME,
            "dataset": DATASET_PATH,
            "lora_path": LORA_PATH,
            "n_questions": len(questions),
            "base_accuracy": round(base_acc, 1),
            "ft_accuracy": round(ft_acc, 1),
            "delta": round(ft_acc - base_acc, 1),
            "mcnemar_b01": b01,
            "mcnemar_b10": b10,
            "mcnemar_p": round(result.pvalue, 4) if b10 + b01 > 0 else None,
            "base_results": base_results,
            "ft_results": ft_results,
        }
        out_path = os.path.join(OUTPUT_DIR, f"eval_{RUN_NAME}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_path}", flush=True)
    else:
        print(f"LORA_PATH not found: {LORA_PATH}, skipping finetuned eval", flush=True)
