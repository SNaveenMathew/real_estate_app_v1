"""Phoenix tracing and Prometheus metrics for General Chat.

Phoenix is used as a local, open-source trace collector/UI. The application
also exposes Prometheus-compatible metrics using the existing
``prometheus-client`` package.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import os
import socket
import subprocess
import sys
from urllib.parse import urlparse
from contextlib import nullcontext
from typing import Any, Iterator

from config import settings

logger = logging.getLogger(__name__)

_setup_lock = threading.Lock()
_tracer_provider = None
_tracer = None
_metrics_ready = False
_phoenix_process = None
_phoenix_started_by_app = False

# Prometheus is already a project dependency, so metrics remain free,
# local, and vendor-neutral.
from prometheus_client import Counter, Histogram

GENERAL_CHAT_REQUESTS = Counter(
    "general_chat_requests_total",
    "Total General Chat requests.",
)
GENERAL_CHAT_ERRORS = Counter(
    "general_chat_errors_total",
    "Total General Chat requests that failed.",
)
GENERAL_CHAT_LATENCY = Histogram(
    "general_chat_request_duration_seconds",
    "General Chat end-to-end request latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
GENERAL_CHAT_LLM_CALLS = Counter(
    "general_chat_llm_calls_total",
    "LLM invocations performed by General Chat.",
)
GENERAL_CHAT_TOOL_CALLS = Counter(
    "general_chat_tool_calls_total",
    "Tool calls performed by General Chat.",
)
GENERAL_CHAT_VALIDATION_BLOCKS = Counter(
    "general_chat_validation_blocks_total",
    "General Chat responses changed by the response validator.",
)
GENERAL_CHAT_REPLY_CHARS = Histogram(
    "general_chat_reply_characters",
    "Characters in final General Chat responses.",
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000),
)


def _phoenix_endpoint_parts() -> tuple[str, int, str]:
    endpoint = settings.phoenix_ui_url.rstrip("/")
    try:
        parsed = urlparse(endpoint)
        return parsed.hostname or "127.0.0.1", parsed.port or 6006, endpoint
    except Exception:
        return "127.0.0.1", 6006, endpoint


def _phoenix_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def ensure_phoenix_server(wait_seconds: float = 15.0) -> bool:
    """Start local Phoenix when needed and wait for its HTTP port.

    This is shared by the FastAPI app and the offline evaluation runner, so
    evaluations do not accidentally configure an OTLP exporter before a
    collector exists. Failure is fail-open: the caller may still run with
    tracing disabled.
    """
    global _phoenix_process, _phoenix_started_by_app

    if not settings.phoenix_enabled:
        return False

    host, port, endpoint = _phoenix_endpoint_parts()
    if _phoenix_port_open(host, port):
        _phoenix_started_by_app = False
        return True

    if _phoenix_process is not None and _phoenix_process.poll() is None:
        return True

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
    if creationflags:
        kwargs["creationflags"] = creationflags

    cmd = [
        sys.executable, "-m", "phoenix.server.main", "serve",
        "--host", host, "--port", str(port),
    ]
    try:
        _phoenix_process = subprocess.Popen(cmd, **kwargs)
        _phoenix_started_by_app = True
        logger.info("Started Phoenix in background: pid=%s endpoint=%s", _phoenix_process.pid, endpoint)
    except Exception:
        _phoenix_process = None
        _phoenix_started_by_app = False
        logger.exception("Could not start Phoenix automatically")
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _phoenix_process.poll() is not None:
            logger.error("Phoenix exited during startup")
            _phoenix_process = None
            _phoenix_started_by_app = False
            return False
        if _phoenix_port_open(host, port):
            logger.info("Phoenix is ready at %s", endpoint)
            return True
        time.sleep(0.2)

    logger.warning("Phoenix did not become ready within %.1fs", wait_seconds)
    return False


def stop_phoenix_server() -> None:
    """Stop only a Phoenix process started by this Python process."""
    global _phoenix_process, _phoenix_started_by_app
    if not _phoenix_started_by_app or _phoenix_process is None:
        return
    proc = _phoenix_process
    _phoenix_process = None
    _phoenix_started_by_app = False
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            logger.debug("Unable to stop Phoenix process", exc_info=True)


def initialize_observability() -> None:
    """Initialize Phoenix OTEL tracing once per Python process.

    Tracing failure is deliberately non-fatal: the chat app must continue to
    work even when the local Phoenix collector is stopped.
    """
    global _tracer_provider, _tracer, _metrics_ready

    if not settings.phoenix_enabled:
        _tracer = False
        _metrics_ready = True
        return

    if _tracer is not None:
        return

    with _setup_lock:
        if _tracer is not None:
            return

        try:
            from phoenix.otel import register

            endpoint = settings.phoenix_collector_endpoint.rstrip("/")
            if not endpoint.endswith("/v1/traces") and settings.phoenix_protocol == "http/protobuf":
                endpoint = f"{endpoint}/v1/traces"

            _tracer_provider = register(
                project_name=settings.phoenix_project_name,
                endpoint=endpoint,
                protocol=settings.phoenix_protocol,
                batch=True,
                auto_instrument=False,
            )
            _tracer = _tracer_provider.get_tracer(
                "real_estate_app.general_chat",
                "1.0.0",
            )
            logger.info(
                "Phoenix observability enabled: project=%s endpoint=%s",
                settings.phoenix_project_name,
                endpoint,
            )
        except Exception:
            logger.exception("Phoenix tracing could not be initialized; continuing without tracing.")
            _tracer = False

        _metrics_ready = True


def phoenix_enabled() -> bool:
    initialize_observability()
    return bool(_tracer and _tracer is not False)


def tracer():
    initialize_observability()
    return _tracer if _tracer and _tracer is not False else None


def trace_span(name: str, *, attributes: dict[str, Any] | None = None):
    """Return a current-span context manager, or a no-op when Phoenix is off."""
    current_tracer = tracer()
    if current_tracer is None:
        return nullcontext(None)

    span = current_tracer.start_span(name)
    if attributes:
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception:
                # A malformed optional attribute must never break a chat turn.
                logger.debug("Unable to set Phoenix attribute %s", key, exc_info=True)
    return _make_current_span_context(span)


class _make_current_span_context:
    """Small context manager that makes a manually created span current."""

    def __init__(self, span):
        self.span = span
        self._cm = None

    def __enter__(self):
        from opentelemetry import trace as trace_api
        self._cm = trace_api.use_span(self.span, end_on_exit=True)
        self._cm.__enter__()
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


def mark_span_error(span, exc: BaseException) -> None:
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        logger.debug("Unable to mark Phoenix span as errored", exc_info=True)


def set_span_output(span, output: Any, *, mime_type: str = "text/plain") -> None:
    if span is None:
        return
    try:
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, default=str)
        span.set_attribute("output.value", output)
        span.set_attribute("output.mime_type", mime_type)
    except Exception:
        logger.debug("Unable to set Phoenix span output", exc_info=True)


def set_span_input(span, value: Any, *, mime_type: str = "text/plain") -> None:
    if span is None:
        return
    try:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        span.set_attribute("input.value", value)
        span.set_attribute("input.mime_type", mime_type)
    except Exception:
        logger.debug("Unable to set Phoenix span input", exc_info=True)


def get_trace_id(span) -> str | None:
    if span is None:
        return None
    try:
        trace_id = span.get_span_context().trace_id
        if not trace_id:
            return None
        return f"{trace_id:032x}"
    except Exception:
        return None


def get_trace_url(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    return f"{settings.phoenix_ui_url.rstrip('/')}/redirects/traces/{trace_id}"


def start_general_chat(message: str, session_id: str | None, history_length: int):
    """Start the root General Chat span and return its context + trace ID."""
    attrs = {
        "openinference.span.kind": "CHAIN",
        "general_chat.message_length": len(message),
        "general_chat.history_length": history_length,
        "session.id": session_id or "",
    }
    context = trace_span("general_chat", attributes=attrs)
    span = context.__enter__()
    set_span_input(span, {"message": message, "history_length": history_length}, mime_type="application/json")
    return context, span, get_trace_id(span), get_trace_url(get_trace_id(span))


def end_general_chat(
    root_span,
    *,
    trace_id: str | None = None,
    reply: str | None = None,
    started_at: float | None = None,
    tool_call_count: int = 0,
    error: BaseException | None = None,
) -> None:
    """Finalize a General Chat trace.

    This is intentionally backward-compatible with the older agent wrapper
    that explicitly finalizes the root span. The newer implementation can
    still manage the context itself; calling this function is safe either way.
    """
    if root_span is not None:
        try:
            root_span.set_attribute("general_chat.tool_call_count", tool_call_count)
            if trace_id:
                root_span.set_attribute("general_chat.trace_id", trace_id)
            if started_at is not None:
                root_span.set_attribute("general_chat.latency_seconds", elapsed(started_at))
            if reply is not None:
                root_span.set_attribute("general_chat.reply_length", len(reply))
                set_span_output(root_span, reply)
            if error is not None:
                mark_span_error(root_span, error)
        except Exception:
            logger.debug("Unable to finalize Phoenix General Chat span", exc_info=True)
    if started_at is not None:
        if error is None:
            GENERAL_CHAT_REQUESTS.inc()
            GENERAL_CHAT_LATENCY.observe(elapsed(started_at))
            GENERAL_CHAT_TOOL_CALLS.inc(tool_call_count)
        else:
            GENERAL_CHAT_REQUESTS.inc()
            GENERAL_CHAT_ERRORS.inc()
            GENERAL_CHAT_LATENCY.observe(elapsed(started_at))


def record_general_chat_success(
    *,
    latency_seconds: float,
    llm_calls: int,
    tool_calls: int,
    reply_chars: int,
    validation_changed: bool,
) -> None:
    GENERAL_CHAT_REQUESTS.inc()
    GENERAL_CHAT_LATENCY.observe(latency_seconds)
    GENERAL_CHAT_LLM_CALLS.inc(llm_calls)
    GENERAL_CHAT_TOOL_CALLS.inc(tool_calls)
    GENERAL_CHAT_REPLY_CHARS.observe(reply_chars)
    if validation_changed:
        GENERAL_CHAT_VALIDATION_BLOCKS.inc()


def record_general_chat_error(latency_seconds: float) -> None:
    GENERAL_CHAT_REQUESTS.inc()
    GENERAL_CHAT_ERRORS.inc()
    GENERAL_CHAT_LATENCY.observe(latency_seconds)


def elapsed(started_at: float) -> float:
    return time.perf_counter() - started_at
