import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and LLM_MODEL) from .env

client = anthropic.Anthropic()

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")

response = client.messages.create(
    model=MODEL,
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}],
)

print(response.content[0].text)
print(f"\nmodel: {response.model}")
print(f"tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
print(f"response type: {type(response).__name__}")
