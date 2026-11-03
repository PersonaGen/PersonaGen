"""OpenAI-compatible chat client helpers."""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from openai import OpenAI


_THREAD_LOCAL = threading.local()


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    return api_base if api_base.endswith("/v1") else f"{api_base}/v1"


def _client(api_base: str, api_key: str) -> OpenAI:
    key = (api_base, api_key)
    if getattr(_THREAD_LOCAL, "client_key", None) != key:
        _THREAD_LOCAL.client = OpenAI(base_url=api_base, api_key=api_key)
        _THREAD_LOCAL.client_key = key
    return _THREAD_LOCAL.client


def chat_completion(
    api_base: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> str:
    client = _client(
        normalize_api_base(api_base),
        api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY") or "EMPTY",
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content
