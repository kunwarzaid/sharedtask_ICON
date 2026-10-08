import json
import re
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_PATH = "./outputs/safety_understanding"

VALID_FILE = (
    "training_data/"
    "safety_understanding_valid.jsonl"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def extract_json(text):

    try:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            return None

        return json.loads(
            match.group(0)
        )

    except Exception:
        return None


# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

true_category = []
pred_category = []

true_persuasion = []
pred_persuasion = []

num_failed = 0

with open(
    VALID_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        item = json.loads(line)

        messages = item["messages"]

        prompt = (
            messages[1]["content"]
        )

        gold = json.loads(
            messages[2]["content"]
        )

        chat = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            chat,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0
        )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        prediction = extract_json(
            decoded
        )

        if prediction is None:

            num_failed += 1
            continue

        true_category.append(
            gold["category"]
        )

        pred_category.append(
            prediction.get(
                "category",
                "UNKNOWN"
            )
        )

        true_persuasion.append(
            gold["persuasion"]
        )

        pred_persuasion.append(
            prediction.get(
                "persuasion",
                "UNKNOWN"
            )
        )

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

print("\n")
print("=" * 80)
print("CATEGORY")
print("=" * 80)

print(
    "Accuracy:",
    accuracy_score(
        true_category,
        pred_category
    )
)

print(
    classification_report(
        true_category,
        pred_category
    )
)

print("\n")
print("=" * 80)
print("PERSUASION")
print("=" * 80)

print(
    "Accuracy:",
    accuracy_score(
        true_persuasion,
        pred_persuasion
    )
)

print(
    classification_report(
        true_persuasion,
        pred_persuasion
    )
)

print("\nFailed JSON parses:", num_failed)
