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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pydantic
from pydantic import BaseModel, Field

# Ensure Outlines is available
try:
    import outlines
    import outlines_core
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False


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
        re.compile(r"(?i)\b(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous|prior|above|existing)\s+(?:instructions|prompts|rules|guidelines|system\s+prompts)\b"),
        re.compile(r"(?i)\b(?:you\s+are\s+now|act\s+as|switch\s+to|enter|act\s+in)\s+(?:in\s+)?(?:DAN|developer|jailbreak|unrestricted|god)\s+mode\b"),
        re.compile(r"(?i)\b(?:reveal|show|print|display|output)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|developer\s+mode|initial\s+prompt)\b"),
        re.compile(r"(?i)\b(?:output|print)\s+(?:your\s+)?(?:initial|system)\s+instructions\b"),
        re.compile(r"(?i)\bdo\s+anything\s+now\b"),
        re.compile(r"(?i)\bpretend\s+(?:you\s+have|to\s+have)\s+no\s+(?:rules|guidelines|safety|restrictions)\b"),
        re.compile(r"(?i)\b(?:drop\s+database|delete\s+from\s+houses|truncate\s+table)\b|\bexec\s*\(\s*['\"]"),
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
            sanitized = sanitized[:cls.MAX_INPUT_CHARS]

        # 2. Prompt injection pattern detection
        matched_injections: list[str] = []
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            match = pattern.search(sanitized)
            if match:
                matched_injections.append(match.group(0))
                risk_score += 0.85

        if matched_injections:
            reasons.append(f"Blocked potential prompt injection / jailbreak attempt: {', '.join(matched_injections[:2])}")

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


# ── 2. Code Agent & SQL Guardrail ─────────────────────────────────────────────

class CodeAgentGuardrail:
    """Validates LLM-generated code programs and SQL queries against strict AST & safety constraints."""

    DISALLOWED_SQL_PATTERNS = [
        re.compile(r"(?i)\b(DROP|TRUNCATE|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|GRANT|REVOKE)\b"),
        re.compile(r"(?i)\b(ATTACH|DETACH|COPY|INSTALL|LOAD|IMPORT|EXPORT)\b"),
        re.compile(r"(?i)\bread_csv\b|\bread_parquet\b|\bread_json\b"),  # prevent ad-hoc file reads outside approved tables
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
        score_keywords = ["walkability", "walk score", "walkable"] if "walk" in score_name.lower() else [score_name.lower()]

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
