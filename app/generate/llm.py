"""vLLM OpenAI-compatible streaming client."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.generate.llm")


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


async def stream_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 600.0,
) -> AsyncIterator[str]:
    """Stream tokens from vLLM /v1/chat/completions. Yields content deltas."""
    settings = get_settings()
    payload = {
        "model": model or settings.vllm_llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.vllm_llm_temperature,
        "max_tokens": max_tokens or settings.vllm_llm_max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(base_url=settings.vllm_llm_url, timeout=timeout) as client:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                body = line.removeprefix("data:").strip()
                if body == "[DONE]":
                    return
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content


async def chat_once(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 120.0,
) -> str:
    settings = get_settings()
    payload = {
        "model": model or settings.vllm_llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.vllm_llm_temperature,
        "max_tokens": max_tokens or 512,
    }
    async with httpx.AsyncClient(base_url=settings.vllm_llm_url, timeout=timeout) as client:
        r = await client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
