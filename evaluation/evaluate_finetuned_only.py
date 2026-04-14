"""
Evaluate ONLY a fine-tuned LoRA on a dataset (no base model evaluation).
Used when the base model has already been evaluated separately (e.g. on Colab).

Strictly identical parameters to evaluate_single.py and evaluate_colab.py
for scientific comparability.
"""
import os
import gc
import json
import re
import torch
from datetime import datetime

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
RUN_NAME = os.environ.get("RUN_NAME", "ft_eval")

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

# Identical to evaluate_single.py and evaluate_colab.py
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


def load_dataset(path):
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def parse_target(item):
    if "target_numeric" in item and item["target_numeric"]:
        return item["target_numeric"][0]
    try:
        return float(item.get("answer", ""))
    except (ValueError, TypeError):
        nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(item.get("answer", "")))
        return float(nums[0]) if nums else None


if __name__ == "__main__":
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"ERROR: DATASET_PATH not set or not found: {DATASET_PATH}")
        import sys; sys.exit(1)
    # LORA_PATH is optional: if empty/none/missing, evaluates base model
    lora_provided = LORA_PATH and LORA_PATH.lower() not in ("none", "")
    if lora_provided and not os.path.exists(LORA_PATH):
        print(f"ERROR: LORA_PATH set but not found: {LORA_PATH}")
        import sys; sys.exit(1)

    questions = load_dataset(DATASET_PATH)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Questions: {len(questions)}")
    print(f"LoRA: {LORA_PATH}")
    print(f"Run name: {RUN_NAME}", flush=True)

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

    if LORA_PATH and LORA_PATH.lower() not in ("none", ""):
        model.load_adapter(LORA_PATH)
        print(f"\nLoaded LoRA from {LORA_PATH}", flush=True)
        eval_label = "FINE-TUNED"
    else:
        print(f"\nEvaluating BASE model (no LoRA)", flush=True)
        eval_label = "BASE"
    FastLanguageModel.for_inference(model)
    print(f"Evaluating {eval_label} model...", flush=True)

    results = []
    for qi, q in enumerate(questions):
        target = parse_target(q)

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
            "question": q["question"][:120],
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
    print(f"\nFINETUNED: {n_correct}/{len(results)} correct ({acc:.1f}%), {n_trunc} truncated", flush=True)

    output = {
        "run_name": RUN_NAME,
        "dataset": DATASET_PATH,
        "lora_path": LORA_PATH,
        "n_questions": len(questions),
        "ft_accuracy": round(acc, 1),
        "ft_results": results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"eval_ft_{RUN_NAME}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}", flush=True)
