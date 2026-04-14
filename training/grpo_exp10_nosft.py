"""
Exp 10: GRPO from scratch (NO SFT warm-start).

Identical to exp7 except:
  - No SFT LoRA loaded — starts from a fresh LoRA on the base Qwen3-8B model
  - Tests whether SFT warm-start helps GRPO or if GRPO can learn from scratch

Pipeline:
  1. Load Qwen3-8B base + fresh LoRA (NO SFT)
  2. GRPO loop with dynamic sampling on 266 pre-screened mixed questions
  3. Checkpoint every SAVE_STEPS useful steps for resume

Reference: DAPO paper (ByteDance/Tsinghua, March 2024)
"""
import os
import gc
import sys
import json
import random
import traceback
import re
import math
import numpy as np
import torch
import torch.nn.functional as F

# ─── PyTorch patch (must be before unsloth import) ────────────────────────────

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

# ── Patch xformers to handle BMGHK backward (Qwen3 GQA) ──────────────────────
# Unsloth's Qwen3Attention_fast_forward calls run_attention → xformers_attention,
# bypassing attn_implementation="eager". xformers can do FORWARD on 5D BMGHK
# but not BACKWARD. We intercept 5D calls and redirect to PyTorch SDPA.
import xformers.ops.fmha as _xf_fmha
_orig_xf_mea = _xf_fmha.memory_efficient_attention

def _patched_memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, **kwargs):
    if query.dim() == 5:
        # BMGHK: (batch, seq, n_groups, heads_per_group, head_dim)
        B, M, G, H, K = query.shape
        # Merge groups+heads → standard 4D: (B, G*H, M, K)
        q = query.reshape(B, M, G * H, K).transpose(1, 2)
        k = key.reshape(B, M, G * H, K).transpose(1, 2)
        v = value.reshape(B, M, G * H, K).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=(attn_bias is not None), dropout_p=p
        )
        return out.transpose(1, 2).reshape(B, M, G, H, K)
    return _orig_xf_mea(query, key, value, attn_bias=attn_bias, p=p, **kwargs)

_xf_fmha.memory_efficient_attention = _patched_memory_efficient_attention
print("Patched xformers BMGHK → PyTorch SDPA fallback", flush=True)

from unsloth import FastLanguageModel

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)



# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 4096  # reduced from 8192 to save VRAM (most reliability questions <2k tokens)

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(SCRIPT_DIR), "data", "dataset_grpo_v4_mixed.json")
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(SCRIPT_DIR, "outputs_exp10")
)

# LoRA config — same as Alex's SFT for fair comparison
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

# GRPO hyperparameters (best from exp1-6bis)
NUM_GENERATIONS       = 4
GEN_MAX_NEW_TOKENS    = 2048  # reduced from 3072 to save VRAM
MAX_USEFUL_STEPS      = int(os.environ.get("MAX_USEFUL_STEPS", "200"))
MAX_TOTAL_STEPS       = int(os.environ.get("MAX_TOTAL_STEPS", "600"))
LEARNING_RATE         = float(os.environ.get("LEARNING_RATE", "1e-5"))
GRAD_ACCUM_STEPS      = 4
MAX_GRAD_NORM         = 0.1
WARMUP_STEPS          = 15
BETA                  = float(os.environ.get("BETA", "0.0"))  # KL penalty
SAVE_STEPS            = 20       # checkpoint every N useful steps
MAX_RESAMPLE_ATTEMPTS = 10       # max re-draws per step before giving up

SYSTEM_PROMPT = """/no_think
You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Rules for your final answer:
- Write ONE single \\boxed{} at the very end of your response — your final answer only.
- Do NOT use \\boxed{} for intermediate steps or calculations.
Always put your single final numerical answer inside \\boxed{}."""


# ─── Regex extraction (identical to all our eval scripts) ─────────────────────

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


# ─── Reward function ──────────────────────────────────────────────────────────

def compute_reward(response_text, target):
    """Compute reward for a single generation. Returns float in [0, 1]."""
    clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    # Penalize multiple boxed answers
    if len(_BOXED_RE.findall(clean)) > 2:
        return -0.3

    preds = extract_boxed_values(clean, last_only=True)
    if not preds:
        return 0.0

    err = abs(preds[0] - target) / (abs(target) + 1e-9)
    if err <= 0.001:  return 1.0
    elif err <= 0.01: return 0.8
    elif err <= 0.05: return 0.4
    else:             return 0.01


# ─── Log-probability computation ─────────────────────────────────────────────

def compute_sequence_log_prob(model, input_ids, response_ids):
    """Compute log P(response | prompt) under the model.

    Args:
        model: the language model
        input_ids: prompt token ids [1, prompt_len]
        response_ids: response token ids [1, response_len]

    Returns:
        sum of log probs over response tokens (scalar tensor)
    """
    full_ids = torch.cat([input_ids, response_ids], dim=1)  # [1, total_len]
    attention_mask = torch.ones_like(full_ids)  # no padding, all tokens attended

    # NOTE: caller must wrap this + backward in allow_mutation_on_saved_tensors
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)

    # logits shape: [1, total_len, vocab_size]
    logits = outputs.logits

    # We want log_prob of each response token given its prefix
    # Shift: logits[t] predicts token[t+1]
    prompt_len = input_ids.shape[1]
    # logits for positions [prompt_len-1, ..., total_len-2] predict tokens [prompt_len, ..., total_len-1]
    relevant_logits = logits[:, prompt_len - 1:-1, :]  # [1, response_len, vocab]
    target_tokens = response_ids  # [1, response_len]

    log_probs = F.log_softmax(relevant_logits, dim=-1)  # [1, response_len, vocab]
    token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)  # [1, response_len]

    return token_log_probs.sum()  # scalar


# ─── Generation helper ────────────────────────────────────────────────────────

def generate_responses(model, tokenizer, prompt_text, n, device):
    """Generate n responses for a prompt. Returns list of (response_text, response_ids)."""
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    results = []

    for gi in range(n):
        torch.manual_seed(SEED + gi * 1000 + random.randint(0, 99999))
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                use_cache=True,
            )
        response_ids = output[:, input_ids.shape[1]:].clone()  # [1, response_len] — clone to free full output
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        results.append((response_text, response_ids, input_ids))

    return results


# ─── Dataset loader ──────────────────────────────────────────────────────────

def load_dataset(path):
    raw = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("answer_type") != "numeric":
                    continue
                try:
                    target = float(item["answer"])
                except (ValueError, TypeError):
                    nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(item.get("answer", "")))
                    target = float(nums[0]) if nums else None
                if target is not None:
                    raw.append({"question": item["question"], "target": target})
    else:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            if "target" in item and item["target"] is not None:
                raw.append({"question": item["question"], "target": float(item["target"])})
            elif "target_numeric" in item and item["target_numeric"]:
                t = item["target_numeric"]
                raw.append({"question": item["question"], "target": float(t[0] if isinstance(t, list) else t)})
            else:
                try:
                    target = float(item.get("answer", ""))
                except (ValueError, TypeError):
                    nums = re.findall(r"-?[\d]+\.?\d*(?:[eE][+-]?\d+)?", str(item.get("answer", "")))
                    target = float(nums[0]) if nums else None
                if target is not None:
                    raw.append({"question": item["question"], "target": target})
    return raw


# ─── Checkpoint save/load ─────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, useful_steps, stats, path):
    os.makedirs(path, exist_ok=True)
    # Save LoRA weights directly in checkpoint dir (compatible with evaluate_finetuned_only.py)
    model.save_pretrained(path)
    # Save optimizer + training state alongside
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "useful_steps": useful_steps,
        "stats": stats,
    }, os.path.join(path, "training_state.pt"))
    print(f"  Checkpoint saved at useful_step={useful_steps} (total_step={step})", flush=True)


# ─── Main training loop ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}"); sys.exit(1)

    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} questions", flush=True)
    print(f"NO SFT — fresh LoRA (r={LORA_R}, alpha={LORA_ALPHA})", flush=True)

    # Load model. Unsloth ignores attn_implementation but we patch xformers
    # at the top of this file to redirect 5D BMGHK calls to PyTorch SDPA.
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create fresh LoRA, save to disk, reload with load_adapter.
    # We can't use get_peft_model directly because Unsloth's fast LoRA kernels
    # create inference tensors incompatible with allow_mutation_on_saved_tensors.
    # load_adapter avoids this by loading into the already-patched model.
    import tempfile
    print(f"Creating fresh LoRA (r={LORA_R}, alpha={LORA_ALPHA})...", flush=True)
    tmp_model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        use_gradient_checkpointing=False,
        random_state=SEED,
    )
    fresh_lora_dir = os.path.join(OUTPUT_DIR, "_fresh_lora_init")
    os.makedirs(fresh_lora_dir, exist_ok=True)
    tmp_model.save_pretrained(fresh_lora_dir)
    print(f"Fresh LoRA saved to {fresh_lora_dir}", flush=True)

    # Reload base model cleanly
    del tmp_model, model
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-apply thinking mode patch
    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_think2(conversation, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig_apply(conversation, **kwargs)
    tokenizer.apply_chat_template = _apply_no_think2

    # Load fresh LoRA via load_adapter (same code path as exp7)
    print(f"Loading fresh LoRA via load_adapter...", flush=True)
    model.load_adapter(fresh_lora_dir, adapter_name="default")
    print("Fresh LoRA loaded (zero-initialized, same path as exp7)", flush=True)

    device = model.device

    # ── Custom training loop setup ──
    # Model stays in EVAL mode to bypass GradientCheckpointLayer's checkpointing
    # path. LoRA gradients work via requires_grad=True (independent of train/eval).
    # In-place ops handled by allow_mutation_on_saved_tensors in the training loop.
    print(f"Model in eval mode, LoRA gradients enabled", flush=True)

    # Enable gradients only on LoRA params
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"LoRA gradients enabled ({sum(p.numel() for p in trainable_params):,} params)", flush=True)

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({len(trainable_params)} tensors)", flush=True)

    # Store reference log-probs per generation (computed before update, reused)
    # We use the "old policy" approach: ref = policy at generation time (GRPO style)
    # This avoids needing a separate ref model copy (saves 50% VRAM)

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.1)

    # Linear warmup + cosine decay scheduler
    # scheduler.step() is called once per optimizer step (every GRAD_ACCUM_STEPS useful steps)
    total_optim_steps = MAX_USEFUL_STEPS // GRAD_ACCUM_STEPS
    warmup_optim_steps = WARMUP_STEPS // GRAD_ACCUM_STEPS

    def lr_lambda(current_step):
        if current_step < warmup_optim_steps:
            return current_step / max(1, warmup_optim_steps)
        progress = (current_step - warmup_optim_steps) / max(1, total_optim_steps - warmup_optim_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Stats tracking
    stats = {
        "useful_steps": 0,
        "wasted_steps": 0,
        "total_attempts": 0,
        "resample_events": 0,
        "losses": [],
        "rewards_per_step": [],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"GRPO Training with Dynamic Sampling", flush=True)
    print(f"  Target useful steps: {MAX_USEFUL_STEPS}", flush=True)
    print(f"  Max total attempts:  {MAX_TOTAL_STEPS}", flush=True)
    print(f"  Max resample/step:   {MAX_RESAMPLE_ATTEMPTS}", flush=True)
    print(f"  Generations/prompt:  {NUM_GENERATIONS}", flush=True)
    print(f"  lr={LEARNING_RATE}, grad_accum={GRAD_ACCUM_STEPS}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Prepare prompt template
    def make_prompt_text(question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    grad_accum_loss = 0.0
    grad_accum_count = 0
    useful_step = 0
    total_step = 0

    try:
        while useful_step < MAX_USEFUL_STEPS and total_step < MAX_TOTAL_STEPS:
            # ── Dynamic sampling: find a prompt with non-zero reward variance ──
            found_useful = False
            for attempt in range(MAX_RESAMPLE_ATTEMPTS):
                total_step += 1
                stats["total_attempts"] += 1

                # Sample a random question
                qi = random.randint(0, len(dataset) - 1)
                item = dataset[qi]
                prompt_text = make_prompt_text(item["question"])
                target = item["target"]

                # Generation phase: inference mode (Unsloth optimized)
                FastLanguageModel.for_inference(model)
                gen_results = generate_responses(model, tokenizer, prompt_text, NUM_GENERATIONS, device)
                gc.collect()
                torch.cuda.empty_cache()
                # Switch back for gradient computation — KEEP eval mode to bypass checkpointing
                FastLanguageModel.for_training(model)
                model.eval()
                # Re-enable LoRA gradients (for_training may have reset them)
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

                # Compute rewards
                rewards = []
                for resp_text, resp_ids, inp_ids in gen_results:
                    r = compute_reward(resp_text, target)
                    rewards.append(r)

                rewards_np = np.array(rewards)

                # Check variance
                if np.std(rewards_np) < 1e-8:
                    # Zero variance — wasted, resample
                    stats["wasted_steps"] += 1
                    n_c = sum(1 for r in rewards if r >= 0.4)
                    tag = "ALL_CORRECT" if n_c == NUM_GENERATIONS else "ALL_WRONG" if n_c == 0 else "SAME_REWARD"
                    print(f"  [attempt {total_step}] SKIP ({tag}, rewards={rewards}) — resampling", flush=True)
                    if attempt < MAX_RESAMPLE_ATTEMPTS - 1:
                        stats["resample_events"] += 1
                    continue

                # Found a useful prompt
                found_useful = True
                break

            if not found_useful:
                print(f"  [step {useful_step}] All {MAX_RESAMPLE_ATTEMPTS} attempts produced zero variance, skipping", flush=True)
                continue

            # ── GRPO update ──────────────────────────────────────────────────

            useful_step += 1
            stats["useful_steps"] = useful_step

            # Compute advantages (group-relative)
            advantages = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)

            # MEMORY-EFFICIENT: backward immediately after each generation's forward.
            # This frees the computation graph for each gen before processing the next,
            # reducing peak memory by ~NUM_GENERATIONS× compared to accumulating all
            # forwards then backward at the end.
            #
            # Each gen contributes 1/NUM_GENERATIONS / GRAD_ACCUM_STEPS to the gradient,
            # so the optimizer.step() (every GRAD_ACCUM_STEPS) gets the same total gradient.
            step_loss_value = 0.0

            for gi, (resp_text, resp_ids, inp_ids) in enumerate(gen_results):
                if resp_ids.shape[1] == 0:
                    continue  # empty response, skip

                advantage = float(advantages[gi])
                if abs(advantage) < 1e-8:
                    continue  # zero advantage, no gradient

                resp_len = resp_ids.shape[1]

                # Forward + backward inside allow_mutation_on_saved_tensors
                # to handle Unsloth's in-place operations
                with torch.autograd.graph.allow_mutation_on_saved_tensors():
                    log_prob = compute_sequence_log_prob(model, inp_ids, resp_ids)
                    # Dr.GRPO objective with per-gen scaling for accumulation
                    gen_loss = -advantage * log_prob / resp_len / NUM_GENERATIONS / GRAD_ACCUM_STEPS
                    gen_loss.backward()

                step_loss_value += gen_loss.item() * GRAD_ACCUM_STEPS  # un-scale for logging

                # Free memory between generations
                del log_prob, gen_loss
                torch.cuda.empty_cache()

            # For logging compatibility
            step_loss = torch.tensor(step_loss_value, device=device)
            scaled_loss = step_loss / GRAD_ACCUM_STEPS

            grad_accum_loss += step_loss.item()
            grad_accum_count += 1

            # Log
            n_correct = sum(1 for r in rewards if r >= 0.4)
            stats["rewards_per_step"].append({
                "useful_step": useful_step,
                "total_step": total_step,
                "question": item["question"][:80],
                "target": target,
                "n_correct": n_correct,
                "rewards": rewards,
                "loss": step_loss.item(),
            })

            print(
                f"[step {useful_step}/{MAX_USEFUL_STEPS}] "
                f"correct={n_correct}/{NUM_GENERATIONS} "
                f"rewards={[f'{r:.2f}' for r in rewards]} "
                f"loss={step_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"wasted={stats['wasted_steps']} "
                f"resampled={stats['resample_events']}",
                flush=True
            )

            # Optimizer step (every GRAD_ACCUM_STEPS)
            if grad_accum_count >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = grad_accum_loss / grad_accum_count
                stats["losses"].append(avg_loss)
                grad_accum_loss = 0.0
                grad_accum_count = 0

            # Checkpoint
            if useful_step % SAVE_STEPS == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{useful_step}")
                save_checkpoint(model, optimizer, scheduler, total_step, useful_step, stats, ckpt_path)

                # Save stats incrementally
                stats_path = os.path.join(OUTPUT_DIR, "training_stats_exp10.json")
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"\nTraining failed at useful_step={useful_step}, total_step={total_step}: {e}", flush=True)
        traceback.print_exc()
        # Emergency checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-emergency-{useful_step}")
        save_checkpoint(model, optimizer, scheduler, total_step, useful_step, stats, ckpt_path)

    # ── Final save ────────────────────────────────────────────────────────────

    # Flush any remaining gradient
    if grad_accum_count > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

    # Save final LoRA
    final_path = os.path.join(OUTPUT_DIR, "lora_final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"\nFinal LoRA saved to: {final_path}", flush=True)

    # Save final stats
    waste_rate = stats["wasted_steps"] / max(1, stats["total_attempts"]) * 100
    stats_path = os.path.join(OUTPUT_DIR, "training_stats_exp10.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}", flush=True)
    print(f"Training complete.", flush=True)
    print(f"  Useful steps: {stats['useful_steps']}", flush=True)
    print(f"  Wasted steps: {stats['wasted_steps']}", flush=True)
    print(f"  Resample events: {stats['resample_events']}", flush=True)
    print(f"  Waste rate: {waste_rate:.1f}%", flush=True)
    print(f"  Total attempts: {stats['total_attempts']}", flush=True)
    print(f"{'='*60}", flush=True)
