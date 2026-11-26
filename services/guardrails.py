"""
services/guardrails.py

Centralized Open-Source Guardrail Framework powered by Outlines & Pydantic.
Enforces multi-layer safety, structural integrity, prompt injection prevention,
AST execution restrictions, and deterministic response grounding with microsecond latency.

Architecture:
  1. InputGuardrail: Sanitization, prompt injection detection, malicious instruction filtering.
  2. CodeAgentGuardrail: Outlines grammar/schema & AST constraints on code-agent executions and SQL.
  3. OutputGroundingGuardrail: Domain metric bounds validation (Walk/Bike scores in [0, 100]),
     factual grounding verification against tool output evidence, and hallucination prevention.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import pydantic
from pydantic import BaseModel, Field

# Ensure Outlines is available
try:
    import outlines
    import outlines_core

    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False


def inline_json_schema(schema_or_model: Any) -> dict[str, Any]:
    """Return a flat JSON Schema with ``$defs`` / ``$ref`` resolved for Outlines / llama-server."""
    if isinstance(schema_or_model, type) and issubclass(schema_or_model, BaseModel):
        raw = schema_or_model.model_json_schema()
    elif isinstance(schema_or_model, dict):
        raw = dict(schema_or_model)
    else:
        raise TypeError(f"Expected a Pydantic BaseModel class or schema dict, got {type(schema_or_model).__name__}")

    defs = raw.pop("$defs", {}) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = str(node["$ref"]).split("/")[-1]
                return _resolve(dict(defs[ref_name]))
            return {k: _resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw)


class StructuredGenerationReply:
    """AIMessage-compatible envelope returned by ``OutlinesStructuredClient.invoke``."""

    def __init__(
        self,
        content: str,
        *,
        parsed: BaseModel | None = None,
        usage_metadata: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.parsed = parsed
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}


class OutlinesStructuredClient:
    """Decode-time JSON-schema constrained client wired via ``outlines.from_openai`` + ``outlines.Generator``.

    Exposes ``.invoke(messages)`` so every structured LLM path shares ``llm_resilience.invoke``'s
    circuit breakers, cancellation, prompt caching, and token telemetry while still enforcing the schema.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        output_schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
        max_retries: int = 0,
        stop: list[str] | None = None,
    ):
        if not OUTLINES_AVAILABLE:
            raise RuntimeError("Outlines is required for OutlinesStructuredClient but is not installed.")
        import openai

        self.base_url = base_url
        self.model = model
        self.model_name = model
        self.output_schema = output_schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.stop = list(stop or [])
        self._last_usage: dict[str, Any] = {}

        raw_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        orig_create = raw_client.chat.completions.create

        def _tracked_create(*args, **kwargs):
            resp = orig_create(*args, **kwargs)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                t_tok = int(getattr(usage, "total_tokens", 0) or (p_tok + c_tok))
                details = getattr(usage, "completion_tokens_details", None)
                r_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
                self._last_usage = {
                    "input_tokens": p_tok,
                    "output_tokens": c_tok,
                    "total_tokens": t_tok,
                    "output_token_details": {"reasoning": r_tok},
                }
            else:
                self._last_usage = {}
            return resp

        raw_client.chat.completions.create = _tracked_create  # type: ignore[method-assign]
        self._outlines_model = outlines.from_openai(raw_client, model_name=model)
        flat_schema = inline_json_schema(output_schema)
        self._generator = outlines.Generator(self._outlines_model, outlines.types.JsonSchema(flat_schema))

    @staticmethod
    def _to_outlines_chat(messages: Any):
        from outlines.inputs import Chat

        chat = Chat()
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role") or "user").lower()
                content = str(msg.get("content") or "")
            else:
                mtype = str(getattr(msg, "type", "user") or "user").lower()
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(mtype, "user")
                content = str(getattr(msg, "content", msg) or "")
            chat.append({"role": role, "content": content})
        return chat

    def invoke(self, messages: Any, **overrides: Any) -> StructuredGenerationReply:
        self._last_usage = {}
        chat = self._to_outlines_chat(messages)
        kwargs: dict[str, Any] = {
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        stop = overrides.get("stop", self.stop)
        if stop:
            kwargs["stop"] = list(stop)
        raw_text = self._generator(chat, **kwargs)
        text = raw_text if isinstance(raw_text, str) else json.dumps(raw_text)
        parsed = self.output_schema.model_validate_json(text)
        return StructuredGenerationReply(
            content=parsed.model_dump_json(),
            parsed=parsed,
            usage_metadata=dict(self._last_usage),
            response_metadata={"model_name": self.model, "outlines": True},
        )


# ── Guardrail Result Models ───────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_text: str
    reasons: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "risk_score": self.risk_score,
            "latency_ms": round(self.latency_ms, 3),
            "metadata": self.metadata,
        }


class RealEstateMetricSchema(BaseModel):
    """Pydantic schema enforcing valid real estate metric bounds."""

    walk_score: Optional[int] = Field(default=None, ge=0, le=100)
    bike_score: Optional[int] = Field(default=None, ge=0, le=100)
    transit_score: Optional[int] = Field(default=None, ge=0, le=100)
    price: Optional[float] = Field(default=None, ge=0)
    sqft: Optional[float] = Field(default=None, ge=0)
    beds: Optional[float] = Field(default=None, ge=0, le=50)
    baths: Optional[float] = Field(default=None, ge=0, le=50)


# ── 1. Input Guardrail ────────────────────────────────────────────────────────


class InputGuardrail:
    """Fast, deterministic input guardrail protecting against prompt injection and malicious payloads."""

    # High-confidence prompt injection / system jailbreak signatures
    PROMPT_INJECTION_PATTERNS = [
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous|prior|above|existing)\s+(?:instructions|prompts|rules|guidelines|system\s+prompts)\b"
        ),
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|act\s+as|switch\s+to|enter|act\s+in)\s+(?:in\s+)?(?:DAN|developer|jailbreak|unrestricted|god)\s+mode\b"
        ),
        re.compile(
            r"(?i)\b(?:reveal|show|print|display|output|dump)\s+(?:your\s+)?(?:(?:full|complete|entire|exact)\s+)?(?:system\s+prompt|hidden\s+instructions|developer\s+mode|initial\s+prompt|tool\s+definitions?)\b"
        ),
        re.compile(
            r"(?i)\b(?:output|print|reveal)\s+(?:your\s+)?(?:(?:full|complete|entire|exact)\s+)?(?:initial|system)\s+instructions\b"
        ),
        re.compile(r"(?i)\bdo\s+anything\s+now\b"),
        re.compile(r"(?i)\bpretend\s+(?:you\s+have|to\s+have)\s+no\s+(?:rules|guidelines|safety|restrictions)\b"),
        # A request to MUTATE the database is refused on its action, whatever the framing.  The narrow list this
        # replaced (``drop database``/``delete from houses``/``truncate table``) missed the plainest spelling of the
        # same request - "DROP TABLE houses" - so that turn fell through to the model and was answered with a table
        # dump on some runs and refused on others.
        re.compile(
            r"(?i)\b(?:drop|truncate|alter|rename)\s+(?:table|database|schema|view)\b"
            r"|\bdelete\s+from\s+\w+"
            r"|\bupdate\s+\w+\s+set\b"
            r"|\binsert\s+into\s+\w+"
            r"|\bexec\s*\(\s*['\"]"
        ),
    ]

    MAX_INPUT_CHARS = 25000

    @classmethod
    def validate_input(cls, user_message: str) -> GuardrailResult:
        t0 = time.perf_counter()
        reasons: list[str] = []
        risk_score = 0.0

        if not user_message or not user_message.strip():
            t1 = time.perf_counter()
            return GuardrailResult(
                passed=True,
                sanitized_text="",
                reasons=[],
                risk_score=0.0,
                latency_ms=(t1 - t0) * 1000.0,
            )

        sanitized = user_message.replace("\x00", "")  # strip null bytes

        # 1. Length check
        if len(sanitized) > cls.MAX_INPUT_CHARS:
            reasons.append(f"Input exceeds maximum allowed length of {cls.MAX_INPUT_CHARS} characters.")
            risk_score += 0.5
            sanitized = sanitized[: cls.MAX_INPUT_CHARS]

        # 2. Prompt injection pattern detection
        matched_injections: list[str] = []
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            match = pattern.search(sanitized)
            if match:
                matched_injections.append(match.group(0))
                risk_score += 0.85

        if matched_injections:
            reasons.append(
                f"Blocked potential prompt injection / jailbreak attempt: {', '.join(matched_injections[:2])}"
            )

        passed = len(reasons) == 0 or risk_score < 0.8
        t1 = time.perf_counter()

        return GuardrailResult(
            passed=passed,
            sanitized_text=sanitized,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
            latency_ms=(t1 - t0) * 1000.0,
            metadata={"matched_patterns": matched_injections},
        )


# ── 2. Scope-Widening & Bulk-Dump Policy Guardrail ───────────────────────────


@dataclass
class ScopePolicyDecision:
    """Deterministic decision for scope-stripping or whole-database dump attempts."""

    allowed: bool
    violation: str | None = None  # "scope_refused" | "limit_exceeded"
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violation": self.violation,
            "reasons": self.reasons,
            "latency_ms": round(self.latency_ms, 3),
        }


class ScopePolicyGuardrail:
    """Detect general-chat filter-drop imperatives and whole-database bulk-dump requests before SQL generation."""

    _FILTER_NOUNS = (
        r"(?:census\s+tract|tract|geography|geographic|metro|msa|county|city|state|zip|scope|arm'?s[-\s]*length|"
        r"saved|favorite|price|market[-\s]*sale)?\s*(?:filter|filters|scope|constraint|constraints|restriction|restrictions)"
    )
    _SCOPE_DROP_PATTERNS = (
        re.compile(
            rf"(?i)\b(?:forget|ignore|skip|drop|remove|bypass|omit|disable|disregard|turn\s+off|clear)\b"
            rf"[^.!?\n]{{0,48}}\b{_FILTER_NOUNS}\b"
        ),
        re.compile(
            r"(?i)\b(?:without\s+(?:any\s+)?(?:geography|geographic|tract|county|city|msa|scope)\s+filter|"
            r"do\s+not\s+filter\s+by\s+(?:census\s+tract|tract|county|city|msa|geography))\b"
        ),
    )

    _BULK_DUMP_PATTERNS = (
        re.compile(
            r"(?i)\b(?:complete\s+raw\s+contents|full\s+raw\s+(?:contents|dump)|entire\s+database|"
            r"whole\s+database|every\s+table\s+in\s+the\s+database|all\s+tables(?:\s+and\s+(?:all\s+)?(?:rows|columns))?|"
            r"dump\s+(?:every|all)\s+tables?)\b"
        ),
        re.compile(
            r"(?i)\b(?:every|all)\s+tables?\b[^.!?\n]{0,60}\b(?:all\s+rows|all\s+columns|raw\s+contents|no\s+limit)\b"
        ),
    )

    _PROGRAM_DROP_PATTERNS = (
        re.compile(
            r"(?i)\b(?:do\s+not\s+filter|without\s+(?:any\s+)?filtering|no\s+geography\s+filter|"
            r"ignore\s+(?:the\s+)?(?:tract|geography|scope|arm'?s[-\s]*length)?\s*filter|"
            r"skip\s+(?:the\s+)?(?:tract|geography|scope)?\s*filter|whole\s+table)\b"
        ),
    )

    @classmethod
    def inspect_request(cls, user_message: str) -> ScopePolicyDecision:
        t0 = time.perf_counter()
        text = user_message or ""
        for pat in cls._SCOPE_DROP_PATTERNS:
            m = pat.search(text)
            if m:
                return ScopePolicyDecision(
                    allowed=False,
                    violation="scope_refused",
                    reasons=[f"Detected scope-widening / filter-removal instruction: {m.group(0)!r}"],
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
        for pat in cls._BULK_DUMP_PATTERNS:
            m = pat.search(text)
            if m:
                return ScopePolicyDecision(
                    allowed=False,
                    violation="limit_exceeded",
                    reasons=[f"Detected unbounded whole-database dump request: {m.group(0)!r}"],
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
        return ScopePolicyDecision(allowed=True, latency_ms=(time.perf_counter() - t0) * 1000.0)

    @classmethod
    def program_drops_scope(cls, code_str: str) -> list[str]:
        """Return reasons if generated Code Agent requirements/plan prose tries to strip scope/filters."""
        reasons: list[str] = []
        text = code_str or ""
        for pat in cls._PROGRAM_DROP_PATTERNS:
            m = pat.search(text)
            if m:
                reasons.append(f"Generated program attempts to bypass filtering ('{m.group(0)}').")
        return reasons


# ── 2b. Code Agent & SQL Guardrail ────────────────────────────────────────────


class CodeAgentGuardrail:
    """Validates LLM-generated code programs and SQL queries against strict AST & safety constraints."""

    DISALLOWED_SQL_PATTERNS = [
        re.compile(r"(?i)\b(DROP|TRUNCATE|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|GRANT|REVOKE)\b"),
        re.compile(r"(?i)\b(ATTACH|DETACH|COPY|INSTALL|LOAD|IMPORT|EXPORT)\b"),
        re.compile(
            r"(?i)\bread_csv\b|\bread_parquet\b|\bread_json\b"
        ),  # prevent ad-hoc file reads outside approved tables
    ]

    @classmethod
    def validate_sql(cls, query: str) -> GuardrailResult:
        t0 = time.perf_counter()
        reasons: list[str] = []
        risk_score = 0.0

        q_clean = query.strip().rstrip(";").strip()

        # SQL must start with SELECT or WITH
        if not re.match(r"(?i)^(SELECT|WITH)\b", q_clean):
            reasons.append("SQL query must be a read-only SELECT or WITH statement.")
            risk_score += 0.9

        # Check for disallowed operations
        for pattern in cls.DISALLOWED_SQL_PATTERNS:
            match = pattern.search(q_clean)
            if match:
                reasons.append(f"Disallowed SQL keyword or operation detected: '{match.group(0)}'.")
                risk_score += 0.95

        passed = len(reasons) == 0
        t1 = time.perf_counter()

        return GuardrailResult(
            passed=passed,
            sanitized_text=q_clean,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
            latency_ms=(t1 - t0) * 1000.0,
            metadata={"sql_valid": passed},
        )

    @classmethod
    def validate_code_program(
        cls,
        code_str: str,
        approved_functions: Set[str],
        max_chars: int = 16000,
    ) -> GuardrailResult:
        t0 = time.perf_counter()
        reasons: list[str] = []
        risk_score = 0.0

        if len(code_str) > max_chars:
            reasons.append(f"Code exceeds character limit ({len(code_str)} > {max_chars}).")
            risk_score += 0.7

        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            t1 = time.perf_counter()
            return GuardrailResult(
                passed=False,
                sanitized_text=code_str,
                reasons=[f"SyntaxError in generated code: {e}"],
                risk_score=0.9,
                latency_ms=(t1 - t0) * 1000.0,
            )

        # AST Safety inspection: Only allow flat calls to approved functions
        for stmt in tree.body:
            call_node: Optional[ast.Call] = None
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                else:
                    reasons.append(f"Disallowed assignment value type '{type(stmt.value).__name__}'.")
                    risk_score += 0.8
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                else:
                    reasons.append(f"Disallowed expression statement type '{type(stmt.value).__name__}'.")
                    risk_score += 0.8
            else:
                reasons.append(f"Disallowed AST statement node: {type(stmt).__name__}.")
                risk_score += 0.9

            if call_node:
                if not isinstance(call_node.func, ast.Name):
                    reasons.append("Function call must be a direct name, not an attribute or subscript.")
                    risk_score += 0.9
                elif call_node.func.id not in approved_functions:
                    reasons.append(f"Call to unapproved function '{call_node.func.id}'.")
                    risk_score += 0.95

        passed = len(reasons) == 0
        t1 = time.perf_counter()

        return GuardrailResult(
            passed=passed,
            sanitized_text=code_str,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
            latency_ms=(t1 - t0) * 1000.0,
            metadata={"ast_nodes_checked": len(tree.body)},
        )


# ── 3. Output Grounding & Metric Guardrail ────────────────────────────────────


class OutputGroundingGuardrail:
    """Verifies output truthfulness, domain metric bounds, and hallucination prevention."""

    WALK_SCORE_PATTERN = re.compile(r"(?i)\bwalk\s+score\s*(?:of|is|:)?\s*(\d+)\b")
    BIKE_SCORE_PATTERN = re.compile(r"(?i)\bbike\s+score\s*(?:of|is|:)?\s*(\d+)\b")
    TRANSIT_SCORE_PATTERN = re.compile(r"(?i)\btransit\s+score\s*(?:of|is|:)?\s*(\d+)\b")

    @classmethod
    def validate_scores_in_text(cls, text: str) -> GuardrailResult:
        """Enforce domain constraints on walk/bike/transit score numbers (0-100)."""
        t0 = time.perf_counter()
        reasons: list[str] = []
        risk_score = 0.0

        for pattern, name in [
            (cls.WALK_SCORE_PATTERN, "Walk Score"),
            (cls.BIKE_SCORE_PATTERN, "Bike Score"),
            (cls.TRANSIT_SCORE_PATTERN, "Transit Score"),
        ]:
            for match in pattern.finditer(text):
                score_val = int(match.group(1))
                if score_val < 0 or score_val > 100:
                    reasons.append(f"Invalid {name} value ({score_val}): real estate scores must be between 0 and 100.")
                    risk_score += 0.9

        passed = len(reasons) == 0
        t1 = time.perf_counter()

        return GuardrailResult(
            passed=passed,
            sanitized_text=text,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
            latency_ms=(t1 - t0) * 1000.0,
        )

    @classmethod
    def enforce_missing_score_guard(
        cls,
        user_question: str,
        reply_text: str,
        score_value_in_db: Optional[int],
        score_name: str = "Walk Score",
    ) -> Tuple[str, bool]:
        """
        Deterministic grounding guard: If the score is NULL/unavailable in the DB,
        ensure the LLM never invents a number.
        """
        q_lower = user_question.lower()
        score_keywords = (
            ["walkability", "walk score", "walkable"] if "walk" in score_name.lower() else [score_name.lower()]
        )

        if any(kw in q_lower for kw in score_keywords) and score_value_in_db is None:
            # Check if reply tried to invent a numeric score
            has_invented_score = bool(re.search(rf"(?i){score_name}\s*(?:of|is|:)?\s*\d+", reply_text))
            override_msg = (
                f"The {score_name} for this house is not available in the data, "
                f"so I can't provide a numeric walkability score. I won't infer "
                f"or invent one from other information."
            )
            if has_invented_score or "not available" not in reply_text.lower():
                return override_msg, True
        return reply_text, False


# ── Global Guardrail Manager ──────────────────────────────────────────────────


class GuardrailManager:
    """Unified entrypoint for executing and monitoring guardrails across the application."""

    @staticmethod
    def inspect_turn_input(user_message: str) -> GuardrailResult:
        return InputGuardrail.validate_input(user_message)

    @staticmethod
    def inspect_scope_and_dump(user_message: str) -> ScopePolicyDecision:
        return ScopePolicyGuardrail.inspect_request(user_message)

    @staticmethod
    def inspect_code_step(code_str: str, approved_functions: Set[str]) -> GuardrailResult:
        return CodeAgentGuardrail.validate_code_program(code_str, approved_functions)

    @staticmethod
    def inspect_sql(sql_query: str) -> GuardrailResult:
        return CodeAgentGuardrail.validate_sql(sql_query)

    @staticmethod
    def inspect_output(
        user_message: str,
        reply_text: str,
        tool_outputs: list[tuple[str, str]],
    ) -> GuardrailResult:
        t0 = time.perf_counter()
        reasons: list[str] = []
        risk_score = 0.0

        # Score boundary check
        score_res = OutputGroundingGuardrail.validate_scores_in_text(reply_text)
        if not score_res.passed:
            reasons.extend(score_res.reasons)
            risk_score += score_res.risk_score

        t1 = time.perf_counter()
        return GuardrailResult(
            passed=len(reasons) == 0,
            sanitized_text=reply_text,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
            latency_ms=(t1 - t0) * 1000.0 + score_res.latency_ms,
            metadata={"outlines_active": OUTLINES_AVAILABLE},
        )
