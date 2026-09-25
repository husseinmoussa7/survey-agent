# llm_openai.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in a .env file (OPENAI_API_KEY=...) "
            "or export it in your shell environment."
        )
    # Optional: allow custom base URL if needed (e.g., enterprise proxy)
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def openai_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Query OpenAI LLM with a prompt. Defaults can be set via environment:
      OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
    """
    client = _get_client()

    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if temperature is None:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    if max_tokens is None:
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
