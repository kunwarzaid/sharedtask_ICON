# common.py
import json

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 1024

SYSTEM_PROMPT = """You are a multilingual safety analysis system.

Given a user prompt, identify:

1. safety category
2. persuasion strategy

Return ONLY valid JSON.

Schema:

{
  "category": "...",
  "persuasion": "..."
}"""

CATEGORIES = [
    "Copyright", "Economic_Harm", "Financial", "Fraud",
    "Gov_Decision", "Hate_Speech", "Privacy_Violence",
]

POSITION_TO_PERSUASION = {
    0: "plain",
    1: "logical_appeal",
    2: "authority_endorsement",
    3: "misrepresentation",
    4: "anchoring",
    5: "priming",
    6: "confirmation_bias",
}

PERSUASIONS = list(POSITION_TO_PERSUASION.values())


def build_prompt(tokenizer, user_prompt):
    """Exact text the model sees before generating. Used in training AND inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # ends with <think>\n\n</think>\n\n
    )


def build_target(category, persuasion):
    return json.dumps(
        {"category": category, "persuasion": persuasion},
        ensure_ascii=False,
    )
