"""Official Gemini generateContent REST client. Does not wrap or invoke `agy`."""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

import httpx

DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAPIError(Exception):
    def __init__(self, status_code: int, message: str, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body

    def openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error" if self.status_code < 500 else "server_error",
                "code": self.status_code,
            }
        }


def gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def gemini_api_base() -> str:
    return (os.environ.get("GEMINI_API_BASE") or DEFAULT_GEMINI_API_BASE).rstrip("/")


def _error_message(status: int, payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
    return fallback or f"Gemini API request failed ({status})"


async def generate_content(
    model: str,
    body: dict[str, Any],
    api_key: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    url = f"{gemini_api_base()}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(url, headers=headers, json=body, timeout=timeout)
        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = {"error": {"message": response.text}}
        if response.status_code >= 400:
            raise GeminiAPIError(
                response.status_code,
                _error_message(response.status_code, payload, response.text),
                payload,
            )
        return payload
    finally:
        if owns_client:
            await client.aclose()


async def stream_generate_content(
    model: str,
    body: dict[str, Any],
    api_key: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 300.0,
) -> AsyncIterator[dict[str, Any]]:
    url = f"{gemini_api_base()}/models/{model}:streamGenerateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    params = {"alt": "sse"}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            params=params,
            json=body,
            timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                raw = await response.aread()
                payload: Any
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
                raise GeminiAPIError(
                    response.status_code,
                    _error_message(response.status_code, payload, raw.decode("utf-8", errors="replace")),
                    payload,
                )
            async for line in response.aiter_lines():
                line = (line or "").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
    finally:
        if owns_client:
            await client.aclose()
