"""Prometheus observability module for Antigravity API Wrapper (agy2api)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    from prometheus_client.exposition import CONTENT_TYPE_LATEST

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    REGISTRY = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

_HTTP_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
_AI_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
_TTFB_BUCKETS = (0.2, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

if PROMETHEUS_AVAILABLE:
    # HTTP layer
    HTTP_REQUESTS_TOTAL = Counter(
        "agy_http_requests_total",
        "Total HTTP requests handled by agy2api",
        ["method", "endpoint", "status_code"],
    )
    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "agy_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint", "status_code"],
        buckets=_HTTP_BUCKETS,
    )
    IN_FLIGHT_REQUESTS = Gauge(
        "agy_in_flight_requests",
        "Number of HTTP requests currently being handled",
        ["endpoint"],
    )

    # Chat completions layer
    CHAT_COMPLETIONS_TOTAL = Counter(
        "agy_chat_completions_total",
        "Total chat completion requests handled",
        ["model", "stream", "status"],
    )
    CHAT_COMPLETION_DURATION_SECONDS = Histogram(
        "agy_chat_completion_duration_seconds",
        "Total duration of chat completion in seconds",
        ["model", "stream"],
        buckets=_AI_BUCKETS,
    )
    CHAT_FIRST_TOKEN_SECONDS = Histogram(
        "agy_chat_first_token_seconds",
        "Time until first token/chunk is emitted in streaming response",
        ["model"],
        buckets=_TTFB_BUCKETS,
    )
    CHAT_TOKENS_TOTAL = Counter(
        "agy_chat_tokens_total",
        "Total tokens processed across models",
        ["model", "token_type"],
    )

    # Antigravity CLI runner execution layer
    RUNNER_EXECUTION_SECONDS = Histogram(
        "agy_runner_execution_seconds",
        "Duration of agy CLI command execution in seconds",
        ["model", "output_format", "status"],
        buckets=_AI_BUCKETS,
    )

    # Image & Speech
    IMAGE_GENERATIONS_TOTAL = Counter(
        "agy_image_generations_total",
        "Total image generation requests handled",
        ["status"],
    )
    SPEECH_REQUESTS_TOTAL = Counter(
        "agy_speech_requests_total",
        "Total TTS speech requests handled",
        ["status"],
    )
else:
    HTTP_REQUESTS_TOTAL = None
    HTTP_REQUEST_DURATION_SECONDS = None
    IN_FLIGHT_REQUESTS = None
    CHAT_COMPLETIONS_TOTAL = None
    CHAT_COMPLETION_DURATION_SECONDS = None
    CHAT_FIRST_TOKEN_SECONDS = None
    CHAT_TOKENS_TOTAL = None
    RUNNER_EXECUTION_SECONDS = None
    IMAGE_GENERATIONS_TOTAL = None
    SPEECH_REQUESTS_TOTAL = None


def record_http_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    if not PROMETHEUS_AVAILABLE or HTTP_REQUESTS_TOTAL is None:
        return
    try:
        m = str(method or "UNKNOWN")
        ep = str(endpoint or "unknown")
        sc = str(status_code)
        HTTP_REQUESTS_TOTAL.labels(method=m, endpoint=ep, status_code=sc).inc()
        if HTTP_REQUEST_DURATION_SECONDS is not None and duration >= 0:
            HTTP_REQUEST_DURATION_SECONDS.labels(method=m, endpoint=ep, status_code=sc).observe(duration)
    except Exception as exc:
        logger.debug("Failed to record http request metric: %s", exc)


def record_chat_completion(
    model: str,
    stream: bool,
    status: str,
    duration: float,
    first_token_duration: float | None = None,
) -> None:
    if not PROMETHEUS_AVAILABLE or CHAT_COMPLETIONS_TOTAL is None:
        return
    try:
        m = str(model or "unknown")
        s = "true" if stream else "false"
        st = str(status or "unknown")
        CHAT_COMPLETIONS_TOTAL.labels(model=m, stream=s, status=st).inc()
        if CHAT_COMPLETION_DURATION_SECONDS is not None and duration >= 0:
            CHAT_COMPLETION_DURATION_SECONDS.labels(model=m, stream=s).observe(duration)
        if CHAT_FIRST_TOKEN_SECONDS is not None and first_token_duration is not None and first_token_duration >= 0:
            CHAT_FIRST_TOKEN_SECONDS.labels(model=m).observe(first_token_duration)
    except Exception as exc:
        logger.debug("Failed to record chat completion metric: %s", exc)


def record_tokens(
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> None:
    if not PROMETHEUS_AVAILABLE or CHAT_TOKENS_TOTAL is None:
        return
    try:
        m = str(model or "unknown")
        if prompt_tokens:
            CHAT_TOKENS_TOTAL.labels(model=m, token_type="prompt").inc(max(0, prompt_tokens))
        if completion_tokens:
            CHAT_TOKENS_TOTAL.labels(model=m, token_type="completion").inc(max(0, completion_tokens))
        if total_tokens:
            CHAT_TOKENS_TOTAL.labels(model=m, token_type="total").inc(max(0, total_tokens))
        if cached_tokens:
            CHAT_TOKENS_TOTAL.labels(model=m, token_type="cached").inc(max(0, cached_tokens))
    except Exception as exc:
        logger.debug("Failed to record tokens metric: %s", exc)


def record_runner_execution(model: str, output_format: str, status: str, duration: float) -> None:
    if not PROMETHEUS_AVAILABLE or RUNNER_EXECUTION_SECONDS is None:
        return
    try:
        m = str(model or "default")
        of = str(output_format or "unknown")
        st = str(status or "unknown")
        if duration >= 0:
            RUNNER_EXECUTION_SECONDS.labels(model=m, output_format=of, status=st).observe(duration)
    except Exception as exc:
        logger.debug("Failed to record runner execution metric: %s", exc)


def record_image_generation(status: str) -> None:
    if not PROMETHEUS_AVAILABLE or IMAGE_GENERATIONS_TOTAL is None:
        return
    try:
        IMAGE_GENERATIONS_TOTAL.labels(status=str(status or "unknown")).inc()
    except Exception as exc:
        logger.debug("Failed to record image generation metric: %s", exc)


def record_speech_request(status: str) -> None:
    if not PROMETHEUS_AVAILABLE or SPEECH_REQUESTS_TOTAL is None:
        return
    try:
        SPEECH_REQUESTS_TOTAL.labels(status=str(status or "unknown")).inc()
    except Exception as exc:
        logger.debug("Failed to record speech metric: %s", exc)


def expose_metrics() -> tuple[bytes, str]:
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not available\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
