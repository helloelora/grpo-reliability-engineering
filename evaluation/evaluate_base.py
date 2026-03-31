"""
Evaluate the base Qwen3-8B model (no LoRA) on all 3 splits.
Saves full responses for review.

Run in parallel with evaluate_finetuned.py.
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
print("Applied torch.Tensor.__mul__ patch", flush=True)

from unsloth import FastLanguageModel

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_DIR, "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_DIR, "results"))

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

MODEL_NAME     = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN    = 8192
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 6144

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response — your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
Always put your single final numerical answer inside \\boxed{}."""

EVAL_SPLITS = {
    "A_training":       os.path.join(DATA_DIR, "eval_A_training.json"),
    "B_hard_holdout":   os.path.join(DATA_DIR, "eval_B_hard_holdout.json"),
    "C_master_holdout": os.path.join(DATA_DIR, "eval_C_master_holdout.json"),
}

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)

_LATEX_SCI_RE = re.compile(
    r"(-?[\d]+\.?\d*)\s*"
    r"(?:\\times|\\cdot|\u00d7|\*)\s*"
    r"10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?",
    re.DOTALL,
)

_FRACTION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def extract_boxed_values(text, last_only=False):
    values = []
    for content in _BOXED_RE.findall(text):
        sci_match = _LATEX_SCI_RE.search(content)
        if sci_match:
            base = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            values.append(base * (10 ** exp))
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
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                val = num / den
                if is_percentage:
                    val /= 100
                values.append(val)
                continue

        nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", normalized)
        for n in nums:
            try:
                val = float(n)
                if is_percentage:
                    val /= 100
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


def parse_target(answer_raw):
    try:
        return float(answer_raw)
    except (ValueError, TypeError):
        nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(answer_raw))
        return float(nums[0]) if nums else None


def evaluate_split(model, tokenizer, split_name, dataset_path, output_path):
    print(f"\nEVALUATING: {split_name}", flush=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions", flush=True)

    results = []
    for qi, q in enumerate(questions):
        question_text = q["question"]

        if "target_numeric" in q and q["target_numeric"]:
            target = q["target_numeric"][0]
        else:
            target = parse_target(q.get("answer", ""))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        completions = []
        for gen_i in range(NUM_GENERATIONS):
            with torch.no_grad():
                output = model.generate(
                    **input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    use_cache=True,
                )
            new_tokens = output[0][input_ids["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            preds = extract_boxed_values(clean, last_only=True)
            last_pred = preds[0] if preds else None
            correct = is_correct(last_pred, target)
            raw_boxed = _BOXED_RE.findall(clean)

            completions.append({
                "generation_id": gen_i,
                "response": response,
                "raw_boxed": [rb[:100] for rb in raw_boxed],
                "extracted_value": last_pred,
                "correct": correct,
                "char_count": len(response),
                "timestamp": datetime.now().isoformat(),
            })

        n_correct = sum(1 for c in completions if c["correct"])
        n_total = len(completions)

        results.append({
            "question_id": qi,
            "question": question_text,
            "answer": q.get("answer", ""),
            "target": target,
            "n_correct": n_correct,
            "n_total": n_total,
            "ratio": n_correct / n_total,
            "completions": completions,
        })

        print(f"  [{qi+1}/{len(questions)}] {n_correct}/{n_total} correct | target={target}", flush=True)

        # Incremental save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    overall_c = sum(r["n_correct"] for r in results)
    overall_t = sum(r["n_total"] for r in results)
    all_correct = sum(1 for r in results if r["ratio"] == 1.0)
    all_wrong   = sum(1 for r in results if r["n_correct"] == 0)

    print(f"\n--- {split_name} RESULTS ---", flush=True)
    print(f"  Accuracy: {overall_c}/{overall_t} ({overall_c/overall_t*100:.1f}%)", flush=True)
    print(f"  All correct: {all_correct}, Partial: {len(results) - all_correct - all_wrong}, All wrong: {all_wrong}", flush=True)

    return {
        "split": split_name,
        "n_questions": len(results),
        "accuracy_pct": round(overall_c / overall_t * 100, 1),
        "all_correct": all_correct,
        "partial": len(results) - all_correct - all_wrong,
        "all_wrong": all_wrong,
    }


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
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
    print("BASE model loaded — no LoRA", flush=True)

    summary = {}
    for split_name, split_path in EVAL_SPLITS.items():
        if not os.path.exists(split_path):
            print(f"SKIP {split_name}: {split_path} not found", flush=True)
            continue
        out_path = os.path.join(OUTPUT_DIR, f"eval_base_{split_name}.json")
        summary[split_name] = evaluate_split(
            model, tokenizer, f"BASE / {split_name}", split_path, out_path
        )

    # Summary
    print("\nBASE MODEL SUMMARY", flush=True)
    for name, s in summary.items():
        print(f"  {name}: {s['accuracy_pct']}% | correct={s['all_correct']} partial={s['partial']} wrong={s['all_wrong']}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "eval_base_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("BASE EVALUATION COMPLETE", flush=True)
