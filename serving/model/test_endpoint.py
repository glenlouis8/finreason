"""
Smoke-test the live vLLM endpoint with one FinQA-style question.
Confirms the served model answers and ends with a 'Final Answer:' line.

    pip install openai
    python test_endpoint.py            # assumes vllm on localhost:8000
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

SYSTEM = ("You are a financial analyst. Reason step-by-step and provide the "
          "final numeric answer on a line starting with 'Final Answer:'.")
QUESTION = ("A company's revenue grew from $4.2 billion in 2019 to $5.7 billion "
            "in 2020. What was the percentage growth in revenue?")

resp = client.chat.completions.create(
    model="glen-louis/finreason-qwen2.5-7b-awq",
    messages=[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": QUESTION}],
    temperature=0.0,
    max_tokens=512,
)
print(resp.choices[0].message.content)
