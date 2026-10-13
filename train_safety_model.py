import os
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from common import MODEL_NAME, MAX_LENGTH

TRAIN_FILE = "training_data_clean/safety_understanding_train.jsonl"
VALID_FILE = "training_data_clean/safety_understanding_valid.jsonl"
OUTPUT_DIR = "outputs/safety_understanding_clean"   # old adapter is kept untouched
BEST_DIR = os.path.join(OUTPUT_DIR, "best_adapter")
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
set_seed(SEED)

# ------------------------------------------------------------
# Data + leakage check
# ------------------------------------------------------------
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

for key in ["intent_group", "family_id"]:
    overlap = set(dataset["train"][key]) & set(dataset["validation"][key])
    if overlap:
        raise RuntimeError(f"Leakage on {key}: {len(overlap)} shared values")
print("Train:", len(dataset["train"]), "| Validation:", len(dataset["validation"]), "| Leakage: none")

keep = {"prompt", "completion"}
dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in keep])

# ------------------------------------------------------------
# Tokenizer + bf16 model (no 4-bit on the H100)
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
model.config.use_cache = False
model.enable_input_require_grads()

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    data_seed=SEED,
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    bf16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=MAX_LENGTH,
    completion_only_loss=True,          # now effective: prompt/completion format
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()

# ------------------------------------------------------------
# Pre-flight checks: masking and truncation
# ------------------------------------------------------------
batch = trainer.data_collator([trainer.train_dataset[0]])
labels = batch["labels"][0]
supervised = labels[labels != -100]
supervised_text = tokenizer.decode(supervised)
print("\nSupervised tokens:", len(supervised))
print("Supervised text:", repr(supervised_text))

if '"category"' not in supervised_text or "<|im_start|>" in supervised_text or len(supervised) > 48:
    raise RuntimeError("Loss mask is wrong: supervision must cover only the JSON answer.")

longest = max(len(x) for x in trainer.train_dataset["input_ids"])
print("Longest tokenized train example:", longest)
if longest >= MAX_LENGTH:
    raise RuntimeError("Some examples hit max_length and may be truncated.")

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
trainer.train()
metrics = trainer.evaluate()
print("\nBest-checkpoint eval metrics:", metrics)

trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
print("Saved:", BEST_DIR)
