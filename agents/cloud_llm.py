import os
from openai import OpenAI

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Cloud LLM disabled (no API key)")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CloudLLMError(Exception):
    pass

def generate(prompt: str, system: str = "") -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            timeout=20
        )
        return r.choices[0].message.content
    except Exception as e:
        raise CloudLLMError(str(e))