import os
import gc
import sys
import json
import random
import traceback
import re
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import TrainerCallback

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
print("Applied torch.Tensor.__mul__ patch (Qwen3 overhead fix)", flush=True)

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


# Paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
STATS_PATH   = os.path.join(SCRIPT_DIR, "live_training_stats.txt")
SAMPLES_PATH = os.path.join(SCRIPT_DIR, "training_samples.json")

_global_step = {"n": 0}
SAVE_SAMPLE_EVERY_N_STEPS = 1

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)


# Monitoring callback
class GRPOStatsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"Training Start: {datetime.now()}\n")
            f.write("Step | reward | correctness | boxed_fmt | loss | grad_norm\n")
            f.write("-" * 100 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        with open(self.output_file, "a", encoding="utf-8") as f:
            step        = state.global_step
            reward      = logs.get("reward",                              float("nan"))
            correctness = logs.get("rewards/reward_correctness",          float("nan"))
            fmt         = logs.get("rewards/reward_boxed_format",         float("nan"))
            loss        = logs.get("loss",                                float("nan"))
            grad_norm   = logs.get("grad_norm",                           float("nan"))
            kl          = logs.get("kl",                                  float("nan"))
            clip_ratio  = logs.get("clipfrac",                            float("nan"))
            f.write(
                f"Step {step:3d} | "
                f"reward={reward:.4f} | "
                f"correctness={correctness:.4f} | "
                f"boxed_fmt={fmt:.4f} | "
                f"loss={loss:.6f} | "
                f"grad_norm={grad_norm:.4f} | "
                f"kl={kl:.4f} | "
                f"clip_ratio={clip_ratio:.4f}\n"
            )


# Configuration
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME  = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(SCRIPT_DIR), "data", "dataset_grpo_combined.json")
)

NUM_GENERATIONS       = 4
MAX_PROMPT_LENGTH     = 1024
MAX_COMPLETION_LENGTH = 6144
GEN_MAX_NEW_TOKENS    = 6144
MAX_STEPS             = 80
LEARNING_RATE         = 5e-6
LORA_R                = 32
LORA_ALPHA            = 32

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response — your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
Always put your single final numerical answer inside \\boxed{}."""


# Reward functions

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)

_LATEX_SCI_RE = re.compile(
    r"(-?[\d]+\.?\d*)\s*"
    r"(?:\\times|\\cdot|\u00d7|\*)\s*"
    r"10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?",
    re.DOTALL
)

_FRACTION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def _extract_boxed_values(text: str, last_only: bool = False) -> list:
    """Extract numeric values from \\boxed{}, handling %, scientific notation,
    fractions, thousands separators, and European decimal commas."""
    values = []
    for content in _BOXED_RE.findall(text):
        sci_match = _LATEX_SCI_RE.search(content)
        if sci_match:
            base = float(sci_match.group(1))
            exp  = int(sci_match.group(2))
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


def reward_correctness(completions, target_numeric, **kwargs):
    """Numeric accuracy of the last \\boxed{} value.
    Tolerance tiers: <=0.1% -> 1.0 / <=1% -> 0.8 / <=5% -> 0.4 / >5% -> 0.01"""
    rewards = []
    for completion, targets_json in zip(completions, target_numeric):
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        clean    = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        targets  = json.loads(targets_json)

        all_boxes = _BOXED_RE.findall(clean)
        n_boxes   = len(all_boxes)

        if n_boxes > 2:
            rewards.append(-0.3)
            continue

        preds = _extract_boxed_values(clean, last_only=True)

        if not preds:
            rewards.append(0.0)
            continue

        scores = []
        for t in targets:
            err = abs(preds[0] - t) / (abs(t) + 1e-9)
            if err <= 0.001:  s = 1.0
            elif err <= 0.01: s = 0.8
            elif err <= 0.05: s = 0.4
            else:             s = 0.01
            scores.append(s)
        rewards.append(float(sum(scores) / len(scores)))
    return rewards


def reward_boxed_format(completions, target_numeric=None, **kwargs):
    """Format enforcement. Penalty -0.5 for missing \\boxed{} or hedging (>2 boxed)."""
    rewards = []
    _global_step["n"] += 1
    step = _global_step["n"]

    sample    = completions[0][0]["content"] if isinstance(completions[0], list) else str(completions[0])
    has_boxed = "\\boxed{" in sample
    truncated = not sample.strip().endswith("}")
    n_boxes_sample = len(_BOXED_RE.findall(sample))

    print(
        f"\n[step={step}] len={len(sample):,} | "
        f"boxed={'yes' if has_boxed else 'NO'} | "
        f"n_boxed={n_boxes_sample} | "
        f"truncated={'YES' if (truncated and not has_boxed) else 'no'} | "
        f"tail={repr(sample[-80:])}",
        flush=True
    )
    with open(STATS_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"  [SAMPLE step={step}] "
            f"len={len(sample):,} | "
            f"n_boxed={n_boxes_sample} | "
            f"boxed={'YES' if has_boxed else 'NO':3s} | "
            f"trunc={'YES' if (truncated and not has_boxed) else 'no':5s} | "
            f"tail={repr(sample[-50:])}\n"
        )

    if step % SAVE_SAMPLE_EVERY_N_STEPS == 0:
        questions = kwargs.get("question",       [None] * len(completions))
        targets   = kwargs.get("target_numeric", [None] * len(completions))
        new_entries = []

        for idx, completion in enumerate(completions):
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
            hb       = "\\boxed{" in response
            tr       = not response.strip().endswith("}")
            n_boxes  = len(_BOXED_RE.findall(response))

            last_val = _extract_boxed_values(
                re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL),
                last_only=True
            )

            target_val = None
            est_correct = None
            if targets[idx] is not None:
                try:
                    t_list = json.loads(targets[idx]) if isinstance(targets[idx], str) else targets[idx]
                    if t_list and last_val:
                        t = t_list[0]
                        err = abs(last_val[0] - t) / (abs(t) + 1e-9)
                        est_correct = err <= 0.05
                        target_val  = t
                except Exception:
                    pass

            new_entries.append({
                "step":            step,
                "completion_id":   idx,
                "question":        questions[idx] if idx < len(questions) else None,
                "target":          target_val,
                "target_raw":      targets[idx] if idx < len(targets) else None,
                "response":        response,
                "char_count":      len(response),
                "has_boxed":       hb,
                "n_boxed":         n_boxes,
                "last_boxed_value": last_val[0] if last_val else None,
                "est_correct":     est_correct,
                "truncated":       tr and not hb,
                "timestamp":       datetime.now().isoformat(),
            })

        existing = []
        if os.path.exists(SAMPLES_PATH):
            try:
                with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        existing.extend(new_entries)
        with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        n_b = len(_BOXED_RE.findall(response))
        if n_b > 2:
            score = -0.5
        elif "\\boxed{" in response:
            score = 0.0
        elif "\\boxed" in response:
            score = -0.1
        else:
            score = -0.5
        rewards.append(float(score))
    return rewards


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    full_ds = Dataset.from_list([{
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": i["question"]}
        ],
        "answer":         i["answer"],
        "target_numeric": json.dumps(i["target_numeric"]),
        "question":       i["question"],
    } for i in raw_data])

    print(f"Dataset: {len(full_ds)} samples", flush=True)

    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    full_ds = full_ds.select(indices)

    gc.collect()
    torch.cuda.empty_cache()

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
    print("Applied enable_thinking=False", flush=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    training_args = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        generation_kwargs={
            "max_new_tokens": GEN_MAX_NEW_TOKENS,
            "do_sample":      True,
            "temperature":    0.8,
            "use_cache":      False,
        },
        mask_truncated_completions=True,
        loss_type="dapo",
        beta=0.001,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        bf16=True,
        output_dir=os.path.join(SCRIPT_DIR, "outputs"),
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correctness, reward_boxed_format],
        args=training_args,
        train_dataset=full_ds,
        callbacks=[GRPOStatsCallback(output_file=STATS_PATH)],
    )

    print(f"Training started, live log: tail -f {STATS_PATH}", flush=True)

    try:
        trainer.train()
    except Exception as e:
        diag = {
            "timestamp":  datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "error":      str(e),
            "traceback":  traceback.format_exc(),
        }
        with open(os.path.join(SCRIPT_DIR, "diag.json"), "w") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        print("Training failed, see diag.json")
        raise

    lora_path = os.path.join(SCRIPT_DIR, "lora_model")
    model.save_pretrained(lora_path)
    print(f"Saved LoRA to: {lora_path}", flush=True)
    print("Training complete.", flush=True)
