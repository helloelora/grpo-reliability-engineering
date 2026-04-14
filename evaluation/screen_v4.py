"""
Screen master_dataset_v4 by generating 4 responses per question with Qwen3-8B base.
Keeps only the questions with mixed correct/wrong outcomes (1/4, 2/4, 3/4).

Splits the dataset into 3 chunks via env var CHUNK_INDEX (0, 1, or 2).
Each chunk handles ~288 questions, fits in 24h.

Usage:
    CHUNK_INDEX=0 python screen_v4.py  # questions 0-288
    CHUNK_INDEX=1 python screen_v4.py  # questions 288-577
    CHUNK_INDEX=2 python screen_v4.py  # questions 577-866
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

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(PROJECT_DIR, "data", "master_dataset_v4.jsonl")
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_DIR, "results"))
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "3"))

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 4096
NUM_GENERATIONS = 4
SEED = 3407

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response - your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
Always put your single final numerical answer inside \\boxed{}."""

# Regex extraction (same as all our eval scripts)
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


def parse_target(item):
    try:
        return float(item.get("answer", ""))
    except (ValueError, TypeError):
        nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(item.get("answer", "")))
        return float(nums[0]) if nums else None


if __name__ == "__main__":
    # Load full dataset
    raw = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("answer_type") == "numeric":
                    target = parse_target(d)
                    if target is not None:
                        raw.append({**d, "_target": target})

    n_total = len(raw)
    chunk_size = (n_total + N_CHUNKS - 1) // N_CHUNKS
    start = CHUNK_INDEX * chunk_size
    end = min(start + chunk_size, n_total)
    chunk = raw[start:end]

    print(f"Total dataset: {n_total} questions", flush=True)
    print(f"Chunk {CHUNK_INDEX}/{N_CHUNKS}: questions {start}-{end} ({len(chunk)} items)", flush=True)
    print(f"Generations per question: {NUM_GENERATIONS}", flush=True)

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

    FastLanguageModel.for_inference(model)

    # Output paths (incremental save)
    out_path = os.path.join(OUTPUT_DIR, f"screen_v4_chunk{CHUNK_INDEX}.json")
    progress_path = os.path.join(OUTPUT_DIR, f"screen_v4_chunk{CHUNK_INDEX}_progress.json")

    # Resume if progress exists
    results = []
    start_qi = 0
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            results = json.load(f).get("results", [])
        start_qi = len(results)
        print(f"Resuming from question {start_qi}", flush=True)

    for qi in range(start_qi, len(chunk)):
        item = chunk[qi]
        target = item["_target"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        generations = []
        for gi in range(NUM_GENERATIONS):
            torch.manual_seed(SEED + qi * 100 + gi)
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
            generations.append({
                "gen_id": gi,
                "extracted_value": last_pred,
                "correct": correct,
            })

        n_correct = sum(1 for g in generations if g["correct"])
        is_mixed = (0 < n_correct < NUM_GENERATIONS)

        results.append({
            "global_index": start + qi,
            "chunk_index": qi,
            "question": item["question"],
            "reasoning": item.get("reasoning", ""),
            "answer": str(item.get("answer", "")),
            "target": target,
            "answer_type": item.get("answer_type"),
            "source": item.get("source", ""),
            "n_correct": n_correct,
            "n_total": NUM_GENERATIONS,
            "is_mixed": is_mixed,
            "generations": generations,
        })

        status = "MIXED" if is_mixed else ("ALL_CORRECT" if n_correct == NUM_GENERATIONS else "ALL_WRONG")
        print(f"  [{qi+1}/{len(chunk)}] {status} ({n_correct}/{NUM_GENERATIONS}) | target={target}", flush=True)

        # Save progress after each question
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "n_done": len(results), "n_total": len(chunk)},
                      f, indent=2, ensure_ascii=False)

    # Final save
    n_mixed = sum(1 for r in results if r["is_mixed"])
    n_all_correct = sum(1 for r in results if r["n_correct"] == NUM_GENERATIONS)
    n_all_wrong = sum(1 for r in results if r["n_correct"] == 0)

    summary = {
        "chunk_index": CHUNK_INDEX,
        "n_chunks": N_CHUNKS,
        "global_start": start,
        "global_end": end,
        "n_questions": len(results),
        "n_mixed": n_mixed,
        "n_all_correct": n_all_correct,
        "n_all_wrong": n_all_wrong,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Chunk {CHUNK_INDEX}: {n_mixed} mixed / {n_all_correct} all-correct / {n_all_wrong} all-wrong", flush=True)
    print(f"Saved to {out_path}", flush=True)
