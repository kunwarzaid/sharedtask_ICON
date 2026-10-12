import json
import re
import os
import torch
import pandas as pd

from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "./outputs/safety_understanding/best_adapter"

VALID_FILE = (
    "training_data/"
    "safety_understanding_valid.jsonl"
)

RESULTS_DIR = "./outputs/safety_understanding"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "eval_predictions_fixed.csv"
)

MAX_NEW_TOKENS = 64


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("Model ready.")


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    # Remove Qwen reasoning if it appears anyway
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # First try direct JSON
    try:

        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    # Then locate a JSON object
    match = re.search(
        r"\{[^{}]*\}",
        text,
        flags=re.DOTALL,
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:
            pass

    return None


# ============================================================
# LOAD DATA
# ============================================================

with open(
    VALID_FILE,
    "r",
    encoding="utf-8",
) as f:

    dataset = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print(
    "Evaluation examples:",
    len(dataset)
)


# ============================================================
# STORAGE
# ============================================================

true_category = []
pred_category = []

true_persuasion = []
pred_persuasion = []

results = []

failed = 0


# ============================================================
# EVALUATION
# ============================================================

for idx, item in enumerate(
    tqdm(
        dataset,
        desc="Evaluating"
    )
):

    messages = item["messages"]

    input_messages = messages[:2]

    gold = json.loads(
        messages[2]["content"]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # disable Qwen3 thinking mode
    # --------------------------------------------------------

    chat = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        chat,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,

            use_cache=True,
        )

    prompt_length = (
        inputs["input_ids"]
        .shape[1]
    )

    generated_tokens = (
        outputs[0][prompt_length:]
    )

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    prediction = extract_json(decoded)

    # --------------------------------------------------------
    # Debug first 10 generations
    # --------------------------------------------------------

    if idx < 10:

        print(
            "\n\n========== EXAMPLE",
            idx,
            "=========="
        )

        print(
            "GOLD:",
            gold
        )

        print(
            "RAW GENERATION:",
            repr(decoded)
        )

        print(
            "PARSED:",
            prediction
        )

    row = {

        "index": idx,

        "gold_category":
            gold["category"],

        "gold_persuasion":
            gold["persuasion"],

        "prediction_text":
            decoded,

        "pred_category": None,

        "pred_persuasion": None,

        "json_parse_failed":
            prediction is None,
    }

    if prediction is None:

        failed += 1

    else:

        category = prediction.get(
            "category",
            "UNKNOWN"
        )

        persuasion = prediction.get(
            "persuasion",
            "UNKNOWN"
        )

        row["pred_category"] = category
        row["pred_persuasion"] = persuasion

        true_category.append(
            gold["category"]
        )

        pred_category.append(
            category
        )

        true_persuasion.append(
            gold["persuasion"]
        )

        pred_persuasion.append(
            persuasion
        )

    results.append(row)

    if (
        (idx + 1) % 25 == 0
        or
        idx + 1 == len(dataset)
    ):

        pd.DataFrame(
            results
        ).to_csv(
            RESULTS_FILE,
            index=False,
        )


# ============================================================
# STATISTICS
# ============================================================

successful = (
    len(dataset) - failed
)

print("\n")
print("=" * 60)
print("JSON GENERATION")
print("=" * 60)

print(
    "Total:",
    len(dataset)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Parse success rate:",
    successful / len(dataset)
)


# ============================================================
# CATEGORY
# ============================================================

print("\n")
print("=" * 60)
print("CATEGORY")
print("=" * 60)

if true_category:

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
            pred_category,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            true_category,
            pred_category
        )
    )

else:

    print(
        "No valid category predictions."
    )


# ============================================================
# PERSUASION
# ============================================================

print("\n")
print("=" * 60)
print("PERSUASION")
print("=" * 60)

if true_persuasion:

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
            pred_persuasion,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            true_persuasion,
            pred_persuasion
        )
    )

else:

    print(
        "No valid persuasion predictions."
    )


print(
    "\nPredictions saved:",
    RESULTS_FILE
)
