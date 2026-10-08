#!/usr/bin/env python3

"""
Train a Qwen3-8B QLoRA adapter for multilingual safety understanding.

Task:
    prompt -> {
        "category": "<category>",
        "persuasion": "<persuasion_type>"
    }

Main design choices:
1. QLoRA instead of full-model fine-tuning.
2. Conversational prompt-completion format.
3. Completion-only loss.
4. Strict data validation before training.
5. Family-disjoint train and validation sets.
6. Deterministic generation-based evaluation.
7. Best checkpoint selected using joint exact match.
8. Validation loss retained as a diagnostic metric.

Expected input files:
    training_data/safety_understanding_train.jsonl
    training_data/safety_understanding_valid.jsonl

Supported input record formats:

A. Existing messages format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "{\"category\": ..., ...}"}
    ],
    "family_id": "optional"
}

B. Prompt-completion format:
{
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "completion": [
        {"role": "assistant", "content": "{\"category\": ..., ...}"}
    ],
    "family_id": "optional"
}
"""

import argparse
import inspect
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"

DEFAULT_TRAIN_FILE = (
    "training_data/safety_understanding_train.jsonl"
)

DEFAULT_VALID_FILE = (
    "training_data/safety_understanding_valid.jsonl"
)

DEFAULT_OUTPUT_DIR = "outputs/safety_understanding_qwen3_8b_qlora"

SYSTEM_PROMPT = (
    "You are a multilingual safety-understanding classifier. "
    "Given a user prompt, identify its safety category and persuasion "
    "strategy. Return exactly one valid JSON object with exactly two "
    'keys: "category" and "persuasion". Do not provide explanations, '
    "reasoning, markdown, or any additional text."
)

REQUIRED_TARGET_KEYS = {
    "category",
    "persuasion",
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ---------------------------------------------------------------------
# JSON and label normalization
# ---------------------------------------------------------------------

def canonical_json(
    category: str,
    persuasion: str,
) -> str:
    """
    Produce a deterministic compact JSON target.
    """

    return json.dumps(
        {
            "category": str(category).strip(),
            "persuasion": str(persuasion).strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_json_object(text: str) -> Optional[Dict[str, Any]\]:
    """
    Parse a model response as a strict JSON object.

    First tries the entire response. If that fails, it extracts the
    first visible {...} block. The exact-output metric will still mark
    extra surrounding text as incorrect.
    """

    if not isinstance(text, str):
        return None

    stripped = text.strip()

    try:
        value = json.loads(stripped)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", stripped, flags=re.DOTALL)

    if match is None:
        return None

    try:
        value = json.loads(match.group(0))

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        return None

    return None


def normalize_target_object(
    value: Dict[str, Any],
) -> Dict[str, str\]:
    """
    Normalize supported target-key variants.
    """

    category = value.get("category")

    persuasion = value.get("persuasion")

    if persuasion is None:
        persuasion = value.get("persuasion_type")

    if category is None or persuasion is None:
        raise ValueError(
            "Target must contain category and persuasion labels."
        )

    return {
        "category": str(category).strip(),
        "persuasion": str(persuasion).strip(),
    }


def parse_target_content(
    content: Any,
) -> Dict[str, str\]:
    """
    Parse an assistant target stored as a JSON string or dictionary.
    """

    if isinstance(content, dict):
        value = content

    elif isinstance(content, str):
        try:
            value = json.loads(content.strip())

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Assistant target is not valid JSON: {content!r}"
            ) from exc

    else:
        raise ValueError(
            f"Unsupported target content type: {type(content)}"
        )

    if not isinstance(value, dict):
        raise ValueError("Assistant target must be a JSON object.")

    return normalize_target_object(value)


# ---------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------

def get_message_content(
    message: Dict[str, Any],
) -> str:
    """
    Extract plain-text message content.
    """

    content = message.get("content", "")

    if not isinstance(content, str):
        raise ValueError(
            "Only plain-text message content is supported."
        )

    return content.strip()


def convert_record(
    example: Dict[str, Any],
) -> Dict[str, Any\]:
    """
    Convert an input record into conversational prompt-completion form.
    """

    result: Dict[str, Any] = {}

    if "prompt" in example and "completion" in example:
        prompt = example["prompt"]
        completion = example["completion"]

        if not isinstance(prompt, list):
            raise ValueError(
                "Conversational prompt must be a list of messages."
            )

        if not isinstance(completion, list):
            raise ValueError(
                "Conversational completion must be a list of messages."
            )

        assistant_messages = [
            message
            for message in completion
            if message.get("role") == "assistant"
        ]

        if len(assistant_messages) != 1:
            raise ValueError(
                "Each example must have exactly one assistant completion."
            )

        target = parse_target_content(
            assistant_messages[0].get("content")
        )

        user_messages = [
            message
            for message in prompt
            if message.get("role") == "user"
        ]

        if len(user_messages) != 1:
            raise ValueError(
                "Each example must have exactly one user message."
            )

        user_text = get_message_content(user_messages[0])

    elif "messages" in example:
        messages = example["messages"]

        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        user_messages = [
            message
            for message in messages
            if message.get("role") == "user"
        ]

        assistant_messages = [
            message
            for message in messages
            if message.get("role") == "assistant"
        ]

        if len(user_messages) != 1:
            raise ValueError(
                "Each example must have exactly one user message."
            )

        if len(assistant_messages) != 1:
            raise ValueError(
                "Each example must have exactly one assistant message."
            )

        user_text = get_message_content(user_messages[0])

        target = parse_target_content(
            assistant_messages[0].get("content")
        )

    else:
        prompt_column = None

        for candidate in ["prompt_text", "text", "input"\]:
            if candidate in example:
                prompt_column = candidate
                break

        category = example.get("category")

        persuasion = example.get(
            "persuasion",
            example.get("persuasion_type"),
        )

        if (
            prompt_column is None
            or category is None
            or persuasion is None
        ):
            raise ValueError(
                "Record must use messages, prompt/completion, or contain "
                "a text column plus category and persuasion labels."
            )

        user_text = str(example[prompt_column]).strip()

        target = {
            "category": str(category).strip(),
            "persuasion": str(persuasion).strip(),
        }

    if not user_text:
        raise ValueError("Prompt is empty.")

    target_text = canonical_json(
        category=target["category"],
        persuasion=target["persuasion"],
    )

    result["prompt"] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    result["completion"] = [
        {
            "role": "assistant",
            "content": target_text,
        }
    ]

    result["prompt_text"] = user_text
    result["category"] = target["category"]
    result["persuasion"] = target["persuasion"]

    if example.get("family_id") is not None:
        result["family_id"] = str(example["family_id"]).strip()
    else:
        result["family_id"] = ""

    for optional_column in [
        "id",
        "language",
        "type",
        "variant_position",
    \]:
        if example.get(optional_column) is not None:
            result[optional_column] = str(
                example[optional_column]
            ).strip()
        else:
            result[optional_column] = ""

    return result


def convert_dataset(
    raw_dataset: Dataset,
    split_name: str,
) -> Dataset:
    """
    Convert all records and report any conversion failure clearly.
    """

    converted_records = []

    for index, example in enumerate(raw_dataset):
        try:
            converted_records.append(
                convert_record(example)
            )

        except Exception as exc:
            raise ValueError(
                f"Invalid {split_name} example at index {index}: {exc}"
            ) from exc

    return Dataset.from_list(converted_records)


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def audit_split(
    dataset: Dataset,
    split_name: str,
) -> None:
    """
    Validate labels, prompts, duplicates, and family structure.
    """

    if len(dataset) == 0:
        raise ValueError(f"{split_name} split is empty.")

    prompts = [
        str(value).strip()
        for value in dataset["prompt_text"]
    ]

    categories = [
        str(value).strip()
        for value in dataset["category"]
    ]

    persuasion_labels = [
        str(value).strip()
        for value in dataset["persuasion"]
    ]

    family_ids = [
        str(value).strip()
        for value in dataset["family_id"]
    ]

    empty_prompt_indices = [
        index
        for index, prompt in enumerate(prompts)
        if not prompt
    ]

    unusually_short = [
        (index, prompt)
        for index, prompt in enumerate(prompts)
        if len(prompt.split()) <= 1
    ]

    duplicate_prompt_count = (
        len(prompts) - len(set(prompts))
    )

    missing_category_count = sum(
        not value for value in categories
    )

    missing_persuasion_count = sum(
        not value for value in persuasion_labels
    )

    nonempty_family_ids = [
        value for value in family_ids if value
    ]

    family_counts = Counter(nonempty_family_ids)

    print(f"\n===== {split_name.upper()} AUDIT =====")
    print(f"Rows                 : {len(dataset):,}")
    print(f"Unique prompts       : {len(set(prompts)):,}")
    print(f"Duplicate prompts    : {duplicate_prompt_count:,}")
    print(f"Categories           : {len(set(categories)):,}")
    print(
        f"Persuasion labels    : "
        f"{len(set(persuasion_labels)):,}"
    )

    if nonempty_family_ids:
        print(
            f"Unique families       : "
            f"{len(family_counts):,}"
        )
        print(
            f"Family-size values    : "
            f"{sorted(set(family_counts.values()))}"
        )
    else:
        print("Unique families       : unavailable")

    print("\nCategory distribution:")
    for label, count in sorted(
        Counter(categories).items()
    ):
        print(f"  {label}: {count}")

    print("\nPersuasion distribution:")
    for label, count in sorted(
        Counter(persuasion_labels).items()
    ):
        print(f"  {label}: {count}")

    if empty_prompt_indices:
        raise ValueError(
            f"{split_name} has empty prompts at indices "
            f"{empty_prompt_indices[:20]}"
        )

    if missing_category_count:
        raise ValueError(
            f"{split_name} has {missing_category_count} "
            "missing category labels."
        )

    if missing_persuasion_count:
        raise ValueError(
            f"{split_name} has {missing_persuasion_count} "
            "missing persuasion labels."
        )

    if unusually_short:
        print("\nWARNING: unusually short prompts:")
        for index, prompt in unusually_short[:20\]:
            print(f"  index={index}, prompt={prompt!r}")

        print(
            "Review these rows before trusting validation metrics."
        )


def check_family_leakage(
    train_dataset: Dataset,
    valid_dataset: Dataset,
) -> None:
    """
    Assert that no non-empty family ID crosses the split boundary.
    """

    train_families = {
        str(value).strip()
        for value in train_dataset["family_id"]
        if str(value).strip()
    }

    valid_families = {
        str(value).strip()
        for value in valid_dataset["family_id"]
        if str(value).strip()
    }

    if not train_families or not valid_families:
        print(
            "\nWARNING: family leakage could not be checked because "
            "family_id is missing from one or both JSONL files."
        )
        return

    overlap = train_families
