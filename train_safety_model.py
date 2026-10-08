from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "Qwen/Qwen3-8B"

dataset = load_dataset(
    "json",
    data_files={
        "train": "training_data/safety_understanding_train.jsonl",
        "validation": "training_data/safety_understanding_valid.jsonl",
    },
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

dataset = dataset.map(tokenize)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

args = TrainingArguments(
    output_dir="outputs/safety_understanding",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
