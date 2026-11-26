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
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from config import settings

logger = logging.getLogger(__name__)

_setup_lock = threading.Lock()
_tracer_provider = None
_tracer = None
_metrics_ready = False
_phoenix_process = None
_phoenix_started_by_app = False
_local_trace_exporter = None

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

HOUSE_CHAT_REQUESTS = Counter(
    "house_chat_requests_total",
    "Total House Chat requests.",
)
HOUSE_CHAT_ERRORS = Counter(
    "house_chat_errors_total",
    "Total House Chat requests that failed.",
)
HOUSE_CHAT_LATENCY = Histogram(
    "house_chat_request_duration_seconds",
    "House Chat end-to-end request latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
HOUSE_CHAT_LLM_CALLS = Counter(
    "house_chat_llm_calls_total",
    "LLM invocations performed by House Chat.",
)
HOUSE_CHAT_TOOL_CALLS = Counter(
    "house_chat_tool_calls_total",
    "Tool calls performed by House Chat.",
)
HOUSE_CHAT_VALIDATION_BLOCKS = Counter(
    "house_chat_validation_blocks_total",
    "House Chat responses changed by the response validator.",
)
HOUSE_CHAT_REPLY_CHARS = Histogram(
    "house_chat_reply_characters",
    "Characters in final House Chat responses.",
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000),
)

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total LLM invocations recorded by role and cache status.",
    ["role", "cache_hit"],
)
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "LLM tokens processed by role and kind (prompt, completion, reasoning).",
    ["role", "kind"],
)
LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Estimated LLM inference cost in USD based on configured rates.",
    ["role"],
)
LLM_CALL_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "Wall latency per LLM call in seconds.",
    ["role"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)


@dataclass
class UsageLedger:
    """Thread-safe per-turn accumulator for LLM tokens, latency, cache hits, and estimated cost."""

    llm_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    llm_latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    by_role: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(
        self,
        *,
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        latency_seconds: float,
        cache_hit: bool,
        estimated_cost_usd: float,
    ) -> None:
        with self._lock:
            self.llm_calls += 1
            if cache_hit:
                self.cache_hits += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.total_tokens += total_tokens
            self.llm_latency_seconds += latency_seconds
            self.estimated_cost_usd += estimated_cost_usd
            bucket = self.by_role.setdefault(
                role or "model",
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                    "estimated_cost_usd": 0.0,
                    "models": [],
                },
            )
            bucket["calls"] += 1
            if cache_hit:
                bucket["cache_hits"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["reasoning_tokens"] += reasoning_tokens
            bucket["total_tokens"] += total_tokens
            bucket["latency_seconds"] += latency_seconds
            bucket["estimated_cost_usd"] += estimated_cost_usd
            if model and model not in bucket["models"]:
                bucket["models"].append(model)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
                "llm_latency_seconds": round(self.llm_latency_seconds, 4),
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "by_role": {
                    role: {
                        **stats,
                        "latency_seconds": round(stats["latency_seconds"], 4),
                        "estimated_cost_usd": round(stats["estimated_cost_usd"], 6),
                        "models": list(stats["models"]),
                    }
                    for role, stats in self.by_role.items()
                },
            }


_USAGE_LEDGER: ContextVar[UsageLedger | None] = ContextVar("usage_ledger", default=None)


def enter_usage_scope():
    """Start a fresh per-turn UsageLedger in the current ContextVar scope and return its reset token."""
    return _USAGE_LEDGER.set(UsageLedger())


def exit_usage_scope(token) -> None:
    """Restore the previous UsageLedger scope."""
    _USAGE_LEDGER.reset(token)


@contextmanager
def usage_scope() -> Iterator[UsageLedger]:
    """Context manager collecting LLM token, latency, and cost usage for a single turn or eval call."""
    ledger = UsageLedger()
    token = _USAGE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _USAGE_LEDGER.reset(token)


def current_usage_snapshot() -> dict[str, Any]:
    """Return the active turn's aggregated LLM token/latency/cost dictionary (empty defaults if outside a scope)."""
    ledger = _USAGE_LEDGER.get()
    return ledger.snapshot() if ledger is not None else UsageLedger().snapshot()


def _estimate_call_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    p_rate = float(getattr(settings, "llm_prompt_cost_per_1m_usd", 0.0) or 0.0)
    c_rate = float(getattr(settings, "llm_completion_cost_per_1m_usd", 0.0) or 0.0)
    return (prompt_tokens * p_rate + completion_tokens * c_rate) / 1_000_000.0


def record_llm_call(
    *,
    role: str,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    total_tokens: int = 0,
    latency_seconds: float = 0.0,
    cache_hit: bool = False,
    error: str | None = None,
) -> None:
    """Record per-call LLM telemetry into the active turn ledger, Prometheus metrics, and active OTEL span."""
    role_key = role or "model"
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(completion_tokens or 0))
    r_tok = max(0, int(reasoning_tokens or 0))
    t_tok = max(0, int(total_tokens or (p_tok + c_tok)))
    lat = max(0.0, float(latency_seconds or 0.0))
    cost = 0.0 if cache_hit else _estimate_call_cost_usd(p_tok, c_tok)

    ledger = _USAGE_LEDGER.get()
    if ledger is not None:
        ledger.add(
            role=role_key,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            reasoning_tokens=r_tok,
            total_tokens=t_tok,
            latency_seconds=lat,
            cache_hit=cache_hit,
            estimated_cost_usd=cost,
        )

    try:
        LLM_CALLS_TOTAL.labels(role=role_key, cache_hit="true" if cache_hit else "false").inc()
        LLM_CALL_LATENCY.labels(role=role_key).observe(lat)
        if p_tok:
            LLM_TOKENS_TOTAL.labels(role=role_key, kind="prompt").inc(p_tok)
        if c_tok:
            LLM_TOKENS_TOTAL.labels(role=role_key, kind="completion").inc(c_tok)
        if r_tok:
            LLM_TOKENS_TOTAL.labels(role=role_key, kind="reasoning").inc(r_tok)
        if cost:
            LLM_COST_USD_TOTAL.labels(role=role_key).inc(cost)
    except Exception:
        logger.debug("Unable to update LLM Prometheus counters", exc_info=True)

    try:
        from opentelemetry import trace as trace_api

        span = trace_api.get_current_span()
        if span is not None and span.is_recording():
            if model:
                span.set_attribute("llm.model_name", model)
            span.set_attribute("llm.role", role_key)
            span.set_attribute("llm.token_count.prompt", p_tok)
            span.set_attribute("llm.token_count.completion", c_tok)
            span.set_attribute("llm.token_count.reasoning", r_tok)
            span.set_attribute("llm.token_count.total", t_tok)
            span.set_attribute("llm.cache_hit", bool(cache_hit))
            span.set_attribute("llm.latency_ms", round(lat * 1000.0, 2))
            if cost:
                span.set_attribute("llm.cost_usd", round(cost, 6))
            attrs: dict[str, Any] = {
                "role": role_key,
                "model": model,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "reasoning_tokens": r_tok,
                "total_tokens": t_tok,
                "latency_ms": round(lat * 1000.0, 2),
                "cache_hit": bool(cache_hit),
            }
            if error:
                attrs["error"] = error
            span.add_event("llm.usage", attributes=attrs)
    except Exception:
        logger.debug("Unable to annotate OTEL span with LLM usage", exc_info=True)


class EvaluationTraceExporter:
    """In-process OTEL exporter used to persist complete eval traces locally."""

    def __init__(self):
        self._lock = threading.Lock()
        self._spans = []

    def export(self, spans):
        with self._lock:
            self._spans.extend(spans)
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS
        except Exception:
            return None

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30000):
        return True

    def clear(self):
        with self._lock:
            self._spans.clear()

    def snapshot(self, trace_id: str):
        with self._lock:
            spans = [s for s in self._spans if get_trace_id(s) == trace_id]
        return spans


def _serialize_span(span) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = getattr(span, "parent", None)
    parent_id = None
    if parent is not None and getattr(parent, "span_id", 0):
        parent_id = f"{parent.span_id:016x}"
    status = getattr(span, "status", None)
    status_code = getattr(getattr(status, "status_code", None), "name", None)
    events = []
    for event in getattr(span, "events", ()) or ():
        events.append(
            {
                "name": event.name,
                "timestamp_ns": getattr(event, "timestamp", None),
                "attributes": dict(event.attributes or {}),
            }
        )
    resource = getattr(span, "resource", None)
    resource_attrs = dict(getattr(resource, "attributes", {}) or {})
    instrumentation = getattr(span, "instrumentation_scope", None)
    return {
        "name": span.name,
        "span_id": f"{ctx.span_id:016x}",
        "trace_id": f"{ctx.trace_id:032x}",
        "parent_span_id": parent_id,
        "start_time_ns": getattr(span, "start_time", None),
        "end_time_ns": getattr(span, "end_time", None),
        "status": status_code,
        "status_description": getattr(status, "description", None),
        "attributes": dict(span.attributes or {}),
        "events": events,
        "resource": resource_attrs,
        "instrumentation_scope": {
            "name": getattr(instrumentation, "name", None),
            "version": getattr(instrumentation, "version", None),
        },
    }


def export_trace(trace_id: str, output_path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write one complete local JSON trace for an evaluation example."""
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spans = _local_trace_exporter.snapshot(trace_id) if _local_trace_exporter else []
    payload = {
        "trace_id": trace_id,
        "exported_at": time.time(),
        "metadata": metadata or {},
        "span_count": len(spans),
        "spans": [
            _serialize_span(span) for span in sorted(spans, key=lambda x: (getattr(x, "start_time", 0) or 0, x.name))
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


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
        sys.executable,
        "-m",
        "phoenix.server.main",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
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
    global _tracer_provider, _tracer, _metrics_ready, _local_trace_exporter

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
                # Export spans from a background thread.  batch=False exported each span SYNCHRONOUSLY inside span.end(),
                # so with the collector stopped every span cost ~7 seconds of connection retries - minutes per chat turn.
                # (The local evaluation capture below is a separate in-memory processor and stays synchronous.)
                batch=settings.phoenix_batch_export,
                auto_instrument=False,
            )
            try:
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor

                _local_trace_exporter = EvaluationTraceExporter()
                _tracer_provider.add_span_processor(
                    SimpleSpanProcessor(_local_trace_exporter),
                    replace_default_processor=False,
                )
            except Exception:
                _local_trace_exporter = None
                logger.exception("Unable to install local evaluation trace exporter")

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


def clear_local_trace_capture() -> None:
    if _local_trace_exporter:
        _local_trace_exporter.clear()


def latest_local_trace_id() -> str | None:
    if not _local_trace_exporter:
        return None
    with _local_trace_exporter._lock:
        spans = list(_local_trace_exporter._spans)
    if not spans:
        return None
    spans.sort(key=lambda s: getattr(s, "end_time", 0) or getattr(s, "start_time", 0) or 0, reverse=True)
    return get_trace_id(spans[0])


def local_trace_span_count(trace_id: str) -> int:
    if not _local_trace_exporter:
        return 0
    return len(_local_trace_exporter.snapshot(trace_id))


def tracer():
    if _tracer and _tracer is not False:
        return _tracer
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


def _finish_root_span_context(
    root_span: Any,
    root_context: Any | None,
    error: BaseException | None,
    chat_type: str,
) -> None:
    if root_context is None:
        if root_span is not None:
            try:
                root_span.end()
            except Exception:
                logger.debug("Unable to end Phoenix %s span", chat_type, exc_info=True)
        return

    try:
        if error is None:
            root_context.__exit__(None, None, None)
        else:
            root_context.__exit__(type(error), error, error.__traceback__)
    except Exception:
        logger.debug("Unable to close Phoenix %s span context", chat_type, exc_info=True)


def _attach_usage_attributes(span, usage: dict[str, Any] | None) -> None:
    if span is None or not usage:
        return
    try:
        span.set_attribute("llm.call_count", int(usage.get("llm_calls", 0)))
        span.set_attribute("llm.cache_hits", int(usage.get("cache_hits", 0)))
        span.set_attribute("llm.token_count.prompt", int(usage.get("prompt_tokens", 0)))
        span.set_attribute("llm.token_count.completion", int(usage.get("completion_tokens", 0)))
        span.set_attribute("llm.token_count.reasoning", int(usage.get("reasoning_tokens", 0)))
        span.set_attribute("llm.token_count.total", int(usage.get("total_tokens", 0)))
        if usage.get("estimated_cost_usd"):
            span.set_attribute("llm.estimated_cost_usd", float(usage["estimated_cost_usd"]))
    except Exception:
        logger.debug("Unable to attach usage attributes to span", exc_info=True)


def end_general_chat(
    root_span: Any,
    *,
    root_context: Any | None = None,
    trace_id: str | None = None,
    reply: str | None = None,
    started_at: float | None = None,
    tool_call_count: int = 0,
    token_usage: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Finalize a General Chat trace.

    This is intentionally backward-compatible with the older agent wrapper
    that explicitly finalizes the root span. The newer implementation can
    still manage the context itself; calling this function is safe either way.
    """
    usage = token_usage if token_usage is not None else current_usage_snapshot()
    if root_span is not None:
        try:
            root_span.set_attribute("general_chat.tool_call_count", tool_call_count)
            _attach_usage_attributes(root_span, usage)
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
    _finish_root_span_context(root_span, root_context, error, "General Chat")
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


def start_house_chat(message: str, house_id: str, history_length: int):
    """Start the root House Chat span and return its context + trace ID."""
    attrs = {
        "openinference.span.kind": "CHAIN",
        "house_chat.message_length": len(message),
        "house_chat.history_length": history_length,
        "house_chat.house_id": house_id,
    }
    context = trace_span("house_chat", attributes=attrs)
    span = context.__enter__()
    set_span_input(
        span,
        {"message": message, "house_id": house_id, "history_length": history_length},
        mime_type="application/json",
    )
    return context, span, get_trace_id(span), get_trace_url(get_trace_id(span))


def end_house_chat(
    root_span: Any,
    *,
    root_context: Any | None = None,
    trace_id: str | None = None,
    reply: str | None = None,
    started_at: float | None = None,
    tool_call_count: int = 0,
    token_usage: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Finalize a House Chat trace."""
    usage = token_usage if token_usage is not None else current_usage_snapshot()
    if root_span is not None:
        try:
            root_span.set_attribute("house_chat.tool_call_count", tool_call_count)
            _attach_usage_attributes(root_span, usage)
            if trace_id:
                root_span.set_attribute("house_chat.trace_id", trace_id)
            if started_at is not None:
                root_span.set_attribute("house_chat.latency_seconds", elapsed(started_at))
            if reply is not None:
                root_span.set_attribute("house_chat.reply_length", len(reply))
                set_span_output(root_span, reply)
            if error is not None:
                mark_span_error(root_span, error)
        except Exception:
            logger.debug("Unable to finalize Phoenix House Chat span", exc_info=True)
    _finish_root_span_context(root_span, root_context, error, "House Chat")
    if started_at is not None:
        if error is None:
            HOUSE_CHAT_REQUESTS.inc()
            HOUSE_CHAT_LATENCY.observe(elapsed(started_at))
            HOUSE_CHAT_TOOL_CALLS.inc(tool_call_count)
        else:
            HOUSE_CHAT_REQUESTS.inc()
            HOUSE_CHAT_ERRORS.inc()
            HOUSE_CHAT_LATENCY.observe(elapsed(started_at))


def record_house_chat_success(
    *,
    latency_seconds: float,
    llm_calls: int,
    tool_calls: int,
    reply_chars: int,
    validation_changed: bool,
) -> None:
    HOUSE_CHAT_REQUESTS.inc()
    HOUSE_CHAT_LATENCY.observe(latency_seconds)
    HOUSE_CHAT_LLM_CALLS.inc(llm_calls)
    HOUSE_CHAT_TOOL_CALLS.inc(tool_calls)
    HOUSE_CHAT_REPLY_CHARS.observe(reply_chars)
    if validation_changed:
        HOUSE_CHAT_VALIDATION_BLOCKS.inc()


def record_house_chat_error(latency_seconds: float) -> None:
    HOUSE_CHAT_REQUESTS.inc()
    HOUSE_CHAT_ERRORS.inc()
    HOUSE_CHAT_LATENCY.observe(latency_seconds)


def elapsed(started_at: float) -> float:
    return time.perf_counter() - started_at
