"""
eval/judge.py

LLM-as-judge scoring for free_text golden examples.

Two backends behind one interface (JudgeBackend):
  - LLMJudge:  calls a real chat model — same ChatOpenAI-compatible pattern
    the app itself uses (see config.judge_llama_server_base_url /
    judge_llama_server_model) — and parses a strict JSON verdict. This is
    what a real run uses.
  - MockJudge: a deterministic, keyword-based stand-in with no network
    dependency. Used only by run_eval.py --mock to smoke-test the pipeline's
    plumbing (fixture -> agent -> scoring -> report) when no model server is
    reachable. Its verdicts are never evidence about actual answer quality —
    every result it produces is labeled "(MOCK)" so it can't be mistaken for
    a real score in a report.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from agents import llm_resilience
from config import LLM_STOP_SEQUENCES, settings
from eval.golden_set import GoldenExample
from eval.scoring import ScoreResult
from services.guardrails import OutlinesStructuredClient


class JudgeVerdict(BaseModel):
    """Strict schema enforced on LLM judge responses via Outlines."""

    verdict: Literal["PASS", "FAIL"]
    reasoning: str = Field(..., min_length=1, max_length=600)


_JUDGE_SYSTEM_PROMPT = """You are grading one answer from a real estate assistant against a rubric.

You will be given the question the user asked, a rubric describing what a
correct and faithful answer must and must not say, any presentation artifacts
rendered alongside the reply (table, chart, or map payloads), and the
assistant's actual reply.

Judge strictly on the rubric. A reply can PASS even if it's phrased very
differently from the rubric, as long as it is factually consistent with it
and doesn't state anything the rubric says it must not state. When a reply
refers to a table, chart, or map shown beside the answer, check the rendered
presentation artifacts section before concluding rows/columns are absent.
A reply that omits a required fact, contradicts the rubric, or fabricates
specifics not supported by the rubric or artifacts should FAIL.

Respond with ONLY a JSON object matching {"verdict": "PASS" or "FAIL", "reasoning": "one or two sentences"}.
"""

_JUDGE_USER_TEMPLATE = """Question: {question}

Rubric: {rubric}
{artifacts_block}
Assistant's reply:
{reply}
"""

# One retry. A judge that emits prose instead of JSON, or that times out once, is a
# transport hiccup; twice is a signal about the judge, and it must not be recorded as the
# agent being wrong. ``INCONCLUSIVE`` is that third state - neither a pass to celebrate nor
# a failure to fix - and run_eval.py counts it separately from PASS/FAIL/ERROR.
_MAX_ATTEMPTS = 2


def summarize_artifacts(artifacts: list[dict] | tuple | None, *, max_chars: int = 2000) -> str:
    """Format presentation artifacts concisely for judge verification."""
    if not artifacts:
        return ""
    parts: list[str] = []
    for idx, art in enumerate(artifacts, start=1):
        if not isinstance(art, dict):
            continue
        kind = art.get("kind") or art.get("type") or "artifact"
        title = art.get("title") or f"Artifact {idx}"
        cols = art.get("columns") or []
        rows = art.get("rows") or []
        summary = f"- [{kind}] title={title!r}"
        if cols:
            summary += f" columns={list(cols)}"
        if rows:
            summary += f" row_count={len(rows)} sample_rows={json.dumps(rows[:5], default=str)}"
        parts.append(summary)
    text = "\nPresentation artifacts displayed beside reply:\n" + "\n".join(parts) + "\n"
    return text[:max_chars]


def _truncate_for_judge(reply: str, *, budget_chars: int) -> str:
    """Bound reply length by preserving head and tail so truncations and endings stay visible."""
    text = reply or ""
    if budget_chars <= 200 or len(text) <= budget_chars:
        return text
    head = int(budget_chars * 0.65)
    tail = budget_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted} characters truncated for judge context bound ...]\n\n{text[-tail:]}"


def _build_judge_prompt(example: GoldenExample, reply: str, artifacts: list[dict] | tuple | None = None) -> str:
    artifacts_block = summarize_artifacts(artifacts)
    overhead = len(example.question or "") + len(example.rubric or "") + len(artifacts_block) + 128
    max_total = int(getattr(settings, "judge_prompt_max_chars", 12000) or 12000)
    reply_budget = max(500, max_total - overhead)
    bounded_reply = _truncate_for_judge(reply, budget_chars=reply_budget)
    return _JUDGE_USER_TEMPLATE.format(
        question=example.question,
        rubric=example.rubric,
        artifacts_block=artifacts_block,
        reply=bounded_reply,
    )


class JudgeBackend:
    def judge(self, example: GoldenExample, reply: str, *, artifacts: list[dict] | tuple = ()) -> ScoreResult:
        raise NotImplementedError


class LLMJudge(JudgeBackend):
    """Calls an Outlines schema-constrained chat model as the judge. Lazily connects on first use."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.judge_llama_server_base_url
        self.model = model or settings.judge_llama_server_model
        self._llm = None

    def _client(self):
        if self._llm is None:
            self._llm = OutlinesStructuredClient(
                base_url=self.base_url,
                api_key="not-needed",
                model=self.model,
                output_schema=JudgeVerdict,
                temperature=0.0,
                max_tokens=int(getattr(settings, "judge_max_tokens", 256) or 256),
                timeout=float(getattr(settings, "judge_timeout_s", 60.0) or 60.0),
                max_retries=0,
                stop=LLM_STOP_SEQUENCES,
            )
        return self._llm

    def judge(self, example: GoldenExample, reply: str, *, artifacts: list[dict] | tuple = ()) -> ScoreResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = _build_judge_prompt(example, reply, artifacts=artifacts)
        meta = {
            "judge_model": self.model,
            "judge_max_tokens": int(getattr(settings, "judge_max_tokens", 256) or 256),
            "judge_prompt_max_chars": int(getattr(settings, "judge_prompt_max_chars", 12000) or 12000),
            "judge_prompt_chars": len(prompt),
        }
        last_problem = "judge produced no verdict"
        inconclusive_kind = "inconclusive_unknown"
        attempts = 0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                response = llm_resilience.invoke(
                    self._client(),
                    [
                        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ],
                    role="judge",
                    breaker_name="judge",
                )
                content = getattr(response, "content", response)
            except Exception as exc:  # transport, circuit-breaker, context, or schema validation failure
                msg = str(exc)
                last_problem = f"judge call failed: {msg}"
                low = msg.lower()
                if "validationerror" in type(exc).__name__.lower() or "json" in low:
                    inconclusive_kind = "inconclusive_schema"
                elif "context" in low and ("length" in low or "token" in low):
                    inconclusive_kind = "inconclusive_context_overflow"
                else:
                    inconclusive_kind = "inconclusive_transport"
                continue

            parsed = _parse_judge_json(content if isinstance(content, str) else str(content))
            if parsed is None:
                last_problem = f"judge did not return parseable JSON: {str(content)[:200]!r}"
                inconclusive_kind = "inconclusive_parse"
                continue

            try:
                validated = JudgeVerdict.model_validate(parsed)
            except Exception:
                last_problem = f"judge returned an unrecognized verdict: {parsed}"
                inconclusive_kind = "inconclusive_schema"
                continue

            return ScoreResult(
                example.id,
                validated.verdict,
                "llm_judge",
                validated.reasoning,
                reply,
                extra={**meta, "judge_attempts": attempts},
            )

        return ScoreResult(
            example.id,
            "INCONCLUSIVE",
            "llm_judge",
            f"{last_problem} (after {attempts} attempt(s))",
            reply,
            extra={**meta, "judge_attempts": attempts, "inconclusive_kind": inconclusive_kind},
        )


def _parse_judge_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            loaded = json.loads(m.group())
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class MockJudge(JudgeBackend):
    """
    Deterministic keyword-based stand-in — NOT a real evaluation. Only checks
    that a reply exists and isn't an obvious bail-out, purely to prove the
    harness's plumbing runs end to end without a model server. See module
    docstring.
    """

    _BAIL_PHRASES = ("i cannot", "i can't help", "i don't have access", "i'm not able to")

    model = "mock (--mock)"

    def judge(self, example: GoldenExample, reply: str, *, artifacts: list[dict] | tuple = ()) -> ScoreResult:
        meta = {"judge_model": self.model}
        if not reply or not reply.strip():
            return ScoreResult(example.id, "FAIL", "llm_judge (MOCK)", "empty reply", reply, extra=meta)
        reply_lower = reply.lower()
        if any(p in reply_lower for p in self._BAIL_PHRASES):
            return ScoreResult(example.id, "FAIL", "llm_judge (MOCK)", "reply looks like a bail-out", reply, extra=meta)
        return ScoreResult(
            example.id,
            "PASS",
            "llm_judge (MOCK)",
            "non-empty, non-bail-out reply — MOCK judge, not a real evaluation",
            reply,
            extra=meta,
        )
