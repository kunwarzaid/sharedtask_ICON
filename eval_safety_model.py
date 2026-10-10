import json
import re
import os
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "./outputs/safety_understanding/best_adapter"
VALID_FILE = "training_data/safety_understanding_valid.jsonl"
RESULTS_DIR = "./outputs/safety_understanding"
RESULTS_FILE = os.path.join(RESULTS_DIR, "eval_predictions.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---------------------------------------------------
# LOAD TOKENIZER AND MODEL
# ---------------------------------------------------
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
print("Loading base model...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Loading trained LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)
model.eval()
print("Model loaded successfully.", flush=True)
# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
# ---------------------------------------------------
# LOAD EVALUATION DATA
# ---------------------------------------------------
with open(VALID_FILE, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]
print(f"Evaluation examples: {len(dataset)}", flush=True)
true_category = []
pred_category = []
true_persuasion = []
pred_persuasion = []
num_failed = 0
results = []
# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
    messages = item["messages"]
    # Assumes message 0 = system, 1 = user, 2 = gold assistant response
    gold = json.loads(messages[2]["content"])
    chat = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    # Decode ONLY newly generated tokens, not the prompt
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]
    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
    prediction = extract_json(decoded)
    row = {
        "index": idx,
        "gold_category": gold.get("category"),
        "pred_category": None,
        "gold_persuasion": gold.get("persuasion"),
        "pred_persuasion": None,
        "prediction_text": decoded,
        "json_parse_failed": prediction is None,
    }
    if prediction is None:
        num_failed += 1
    else:
        row["pred_category"] = prediction.get("category", "UNKNOWN")
        row["pred_persuasion"] = prediction.get("persuasion", "UNKNOWN")
        true_category.append(gold["category"])
        pred_category.append(row["pred_category"])
        true_persuasion.append(gold["persuasion"])
        pred_persuasion.append(row["pred_persuasion"])
    results.append(row)
    # Save every 25 examples so progress is not lost
    if (idx + 1) % 25 == 0 or idx + 1 == len(dataset):
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------
print("\n" + "=" * 60)
print("CATEGORY")
print("=" * 60)
if true_category:
    print("Accuracy:", accuracy_score(true_category, pred_category))
    print(classification_report(
        true_category, pred_category, zero_division=0
    ))
else:
    print("No valid JSON predictions to evaluate.")
print("\n" + "=" * 60)
print("PERSUASION")
print("=" * 60)
if true_persuasion:
    print("Accuracy:", accuracy_score(true_persuasion, pred_persuasion))
    print(classification_report(
        true_persuasion, pred_persuasion, zero_division=0
    ))
else:
    print("No valid JSON predictions to evaluate.")
print("\nTotal examples:", len(dataset))
print("Failed JSON parses:", num_failed)
print("Successful JSON parses:", len(dataset) - num_failed)
print("Predictions saved to:", RESULTS_FILE)
