import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Exercise 3: a deliberately wrong key, to see what an auth failure looks like
client = anthropic.Anthropic(api_key="sk-ant-this-key-is-wrong")

try:
    client.messages.create(
        model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=256,
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}],
    )
except anthropic.APIStatusError as e:
    print(f"error class: {type(e).__name__}")
    print(f"HTTP status: {e.status_code}")
    print(f"message:     {e.message}")
