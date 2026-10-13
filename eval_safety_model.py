import json
import os

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import MODEL_NAME, CATEGORIES, PERSUASIONS, build_prompt

ADAPTER_DIR = "outputs/safety_understanding_clean/best_adapter"
VALID_FILE = "training_data_clean/safety_understanding_valid.jsonl"
RESULTS_FILE = "outputs/safety_understanding_clean/validation_predictions.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
device = next(model.parameters()).device

with open(VALID_FILE, encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]
print("Validation examples:", len(data))


def parse(text):
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


rows = []
for i, ex in enumerate(tqdm(data, desc="Evaluating")):
    prompt = build_prompt(tokenizer, ex["user_prompt"])   # same function as training
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape], skip_special_tokens=True).strip()
    pred = parse(raw)

    pred_cat = pred.get("category", "INVALID") if pred else "PARSE_FAILED"
    pred_per = pred.get("persuasion", "INVALID") if pred else "PARSE_FAILED"

    rows.append({
        "family_id": ex["family_id"],
        "language": ex["language"],
        "gold_category": ex["category"],
        "pred_category": pred_cat,
        "gold_persuasion": ex["persuasion"],
        "pred_persuasion": pred_per,
        "json_ok": pred is not None,
        "raw": raw,
    })

    if i < 5:
        print(f"\n[{i}] gold=({ex['category']}, {ex['persuasion']}) raw={raw!r}")

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

df["cat_ok"] = df["gold_category"] == df["pred_category"]
df["per_ok"] = df["gold_persuasion"] == df["pred_persuasion"]
df["joint_ok"] = df["cat_ok"] & df["per_ok"]

print("\n" + "=" * 60)
print(f"JSON validity:      {df['json_ok'].mean():.4f}")
print(f"Out-of-schema labels: "
      f"{(~df['pred_category'].isin(CATEGORIES)).sum()} category, "
      f"{(~df['pred_persuasion'].isin(PERSUASIONS)).sum()} persuasion")
print(f"Category accuracy:  {df['cat_ok'].mean():.4f} | macro-F1 "
      f"{f1_score(df['gold_category'], df['pred_category'], labels=CATEGORIES, average='macro', zero_division=0):.4f}")
print(f"Persuasion accuracy:{df['per_ok'].mean():.4f} | macro-F1 "
      f"{f1_score(df['gold_persuasion'], df['pred_persuasion'], labels=PERSUASIONS, average='macro', zero_division=0):.4f}")
print(f"Joint accuracy:     {df['joint_ok'].mean():.4f}")

print("\nPer language:")
print(df.groupby("language")[["json_ok", "cat_ok", "per_ok", "joint_ok"]].mean().round(4))

print("\nPer category (category accuracy):")
print(df.groupby("gold_category")["cat_ok"].mean().round(4))

consistency = df.groupby("family_id")["pred_category"].nunique().eq(1).mean()
print(f"\nFamilies with one consistent predicted category across 7 variants: {consistency:.4f}")

print("\nCategory report:")
print(classification_report(df["gold_category"], df["pred_category"], labels=CATEGORIES, zero_division=0, digits=4))
print("Category confusion (rows=gold, cols=pred):", CATEGORIES)
print(confusion_matrix(df["gold_category"], df["pred_category"], labels=CATEGORIES))

print("\nPersuasion report:")
print(classification_report(df["gold_persuasion"], df["pred_persuasion"], labels=PERSUASIONS, zero_division=0, digits=4))
print("Persuasion confusion (rows=gold, cols=pred):", PERSUASIONS)
print(confusion_matrix(df["gold_persuasion"], df["pred_persuasion"], labels=PERSUASIONS))

print("\nSaved:", RESULTS_FILE)
