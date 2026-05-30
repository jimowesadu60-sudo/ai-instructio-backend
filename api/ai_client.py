"""Async HTTP adapter for OpenAI-compatible APIs.

Loads ``.env`` from the project root (via python-dotenv). Environment variables:

- Shared (``api_stage`` in ``"api1"``, ``"api2"``, ``"api3"``): prefer
  ``API_KEY``, ``MODEL``, optional ``API_BASE_URL``; if unset, fall back to
  ``API1_API_KEY``, ``API1_MODEL_NAME``, optional ``API1_BASE_URL``.
- Default (no ``api_stage``): ``AI_API_KEY``, ``AI_MODEL_NAME``, optional ``AI_BASE_URL``.

No business logic and no local prompt file I/O.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

# Project root: parent of the ``api`` package directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class AIClientError(Exception):
    """Raised when the AI request fails (timeout, transport, or API error)."""


def _data_url_for_image_base64(item: str, default_mime: str) -> str:
    if item.startswith("data:"):
        return item
    return f"data:{default_mime};base64,{item}"


def _shared_openai_credentials() -> tuple[str, str, Optional[str]]:
    """Credentials for API1–API3: unified API_KEY / MODEL / API_BASE_URL or API1_* fallback."""
    api_key = (os.getenv("API_KEY") or os.getenv("API1_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("API_KEY (or API1_API_KEY) is not set")
    model = (os.getenv("MODEL") or os.getenv("API1_MODEL_NAME") or "").strip()
    if not model:
        raise ValueError("MODEL (or API1_MODEL_NAME) is not set")
    base_raw = (os.getenv("API_BASE_URL") or os.getenv("API1_BASE_URL") or "").strip()
    base_url = base_raw if base_raw else None
    return api_key, model, base_url


class AIClient:
    """Thin async wrapper around ``AsyncOpenAI`` chat completions."""

    def __init__(self, *, api_stage: Optional[str] = None) -> None:
        if api_stage in ("api1", "api2", "api3"):
            api_key, model, base_url = _shared_openai_credentials()
        else:
            api_key = os.getenv("AI_API_KEY")
            if not api_key:
                raise ValueError("AI_API_KEY is not set")

            model = os.getenv("AI_MODEL_NAME")
            if not model:
                raise ValueError("AI_MODEL_NAME is not set")

            base_url_raw = os.getenv("AI_BASE_URL")
            base_url = base_url_raw if base_url_raw else None

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    async def request_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._complete(
            messages, response_format=response_format, model=model
        )

    async def request_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64_list: List[str],
        *,
        default_image_mime: str = "image/png",
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]
        for raw in image_base64_list:
            url = _data_url_for_image_base64(raw, default_image_mime)
            user_content.append(
                {"type": "image_url", "image_url": {"url": url}},
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self._complete(
            messages, response_format=response_format, model=model
        )

    async def _complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        try:
            kwargs: Dict[str, Any] = {
                "model": model if model else self._model,
                "messages": messages,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            completion = await self._client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise AIClientError("AI request timed out") from e
        except (APIError, APIConnectionError, RateLimitError) as e:
            raise AIClientError(str(e)) from e

        choice = completion.choices[0] if completion.choices else None
        if choice is None or choice.message.content is None:
            return ""
        return choice.message.content
