# run_qwen_baseline.py

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INPUT_FILE = "processed/baseline_eval_100.csv"

MODEL_NAME = "Qwen/Qwen3-8B"

df = pd.read_csv(INPUT_FILE)

llm = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
)

prompts = []

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

for prompt in df["prompt"]:


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    prompts.append(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

outputs = llm.generate(
    prompts,
    sampling_params,
)

responses = []

for output in outputs:

    responses.append(
        output.outputs[0].text
    )

df["response"] = responses

df.to_csv(
    "processed/qwen_baseline_results.csv",
    index=False
)

print("Done")
