import json
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3-8B"

TRAIN_FILE = "training_data/safety_understanding_train.jsonl"
VALID_FILE = "training_data/safety_understanding_valid.jsonl"

OUTPUT_DIR = "outputs/safety_understanding"


# =====================================================
# Load Dataset
# =====================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALID_FILE,
    }
)

print(dataset)

# =====================================================
# Tokenizer
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# QLoRA Config
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# =====================================================
# Base Model
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

# =====================================================
# LoRA
# =====================================================

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# =====================================================
# Training
# =====================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    learning_rate=1e-4,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    gradient_accumulation_steps=8,

    bf16=True,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=100,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",

    max_length=512,

    completion_only_loss=True,

    gradient_checkpointing=True,
)

# =====================================================
# Trainer
# =====================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.model.print_trainable_parameters()

trainer.train()

trainer.save_model(
    f"{OUTPUT_DIR}/best_adapter"
)

tokenizer.save_pretrained(
    f"{OUTPUT_DIR}/best_adapter"
)
