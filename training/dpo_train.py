import os
import gc
import sys
import json
import random
import traceback
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset

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
from trl import DPOConfig, DPOTrainer


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if "WORKDIR" in os.environ:
    hf_cache = os.path.join(os.environ["WORKDIR"], ".cache", "huggingface")
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)


# Configuration
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME  = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192
LORA_R      = 32
LORA_ALPHA  = 32

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(SCRIPT_DIR), "data", "dataset_dpo_pairs.json")
)

NUM_EPOCHS    = 3
LEARNING_RATE = 5e-6
BETA          = 0.1  # DPO temperature - controls how strongly to prefer chosen over rejected


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Build HF dataset with prompt/chosen/rejected format
    ds = Dataset.from_list([{
        "prompt": item["prompt"],
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    } for item in raw_data])

    print(f"Dataset: {len(ds)} preference pairs", flush=True)

    # Shuffle
    indices = list(range(len(ds)))
    random.shuffle(indices)
    ds = ds.select(indices)

    gc.collect()
    torch.cuda.empty_cache()

    # Load model
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

    training_args = DPOConfig(
        beta=BETA,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=1024,
        bf16=True,
        output_dir=os.path.join(SCRIPT_DIR, "outputs_dpo"),
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("DPO training started", flush=True)

    try:
        trainer.train()
    except Exception as e:
        diag = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        with open(os.path.join(SCRIPT_DIR, "diag_dpo.json"), "w") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        print("Training failed, see diag_dpo.json")
        raise

    lora_path = os.path.join(SCRIPT_DIR, "lora_model_dpo")
    model.save_pretrained(lora_path)
    print(f"Saved LoRA to: {lora_path}", flush=True)
    print("DPO training complete.", flush=True)
