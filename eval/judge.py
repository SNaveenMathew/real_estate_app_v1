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

from eval.golden_set import GoldenExample
from eval.scoring import ScoreResult


_JUDGE_SYSTEM_PROMPT = """You are grading one answer from a real estate assistant against a rubric.

You will be given the question the user asked, a rubric describing what a
correct and faithful answer must and must not say, and the assistant's
actual reply.

Judge strictly on the rubric. A reply can PASS even if it's phrased very
differently from the rubric, as long as it is factually consistent with it
and doesn't state anything the rubric says it must not state. A reply that
omits a required fact, contradicts the rubric, or fabricates specifics not
supported by it should FAIL.

Respond with ONLY a JSON object, no other text:
{"verdict": "PASS" or "FAIL", "reasoning": "one or two sentences"}
"""

_JUDGE_USER_TEMPLATE = """Question: {question}

Rubric: {rubric}

Assistant's reply:
{reply}
"""

# One retry. A judge that emits prose instead of JSON, or that times out once, is a
# transport hiccup; twice is a signal about the judge, and it must not be recorded as the
# agent being wrong. ``INCONCLUSIVE`` is that third state - neither a pass to celebrate nor
# a failure to fix - and run_eval.py counts it separately from PASS/FAIL/ERROR.
_MAX_ATTEMPTS = 2


class JudgeBackend:
    def judge(self, example: GoldenExample, reply: str) -> ScoreResult:
        raise NotImplementedError


class LLMJudge(JudgeBackend):
    """Calls a real chat model as the judge. Lazily connects on first use."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        from config import settings
        self.base_url = base_url or settings.judge_llama_server_base_url
        self.model = model or settings.judge_llama_server_model
        self._llm = None

    def _client(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                base_url=self.base_url, api_key="not-needed",
                model=self.model, temperature=0.0,
            )
        return self._llm

    def judge(self, example: GoldenExample, reply: str) -> ScoreResult:
        from langchain_core.messages import SystemMessage, HumanMessage
        prompt = _JUDGE_USER_TEMPLATE.format(
            question=example.question, rubric=example.rubric, reply=reply)
        meta = {"judge_model": self.model}
        last_problem = "judge produced no verdict"
        attempts = 0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                response = self._client().invoke([
                    SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                content = response.content
            except Exception as e:                       # transport or server problem
                last_problem = f"judge call failed: {e}"
                continue

            parsed = _parse_judge_json(content)
            if parsed is None:
                last_problem = f"judge did not return parseable JSON: {content[:200]!r}"
                continue

            verdict = str(parsed.get("verdict", "")).upper()
            if verdict not in ("PASS", "FAIL"):
                last_problem = f"judge returned an unrecognized verdict: {parsed}"
                continue
            return ScoreResult(example.id, verdict, "llm_judge", parsed.get("reasoning", ""), reply,
                               extra={**meta, "judge_attempts": attempts})

        # Retried and still unusable: this says nothing about the answer, so it is not an
        # ERROR (which would read as a broken agent) and certainly not a FAIL.
        return ScoreResult(example.id, "INCONCLUSIVE", "llm_judge",
                           f"{last_problem} (after {attempts} attempt(s))", reply,
                           extra={**meta, "judge_attempts": attempts})


def _parse_judge_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
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

    def judge(self, example: GoldenExample, reply: str) -> ScoreResult:
        meta = {"judge_model": self.model}
        if not reply or not reply.strip():
            return ScoreResult(example.id, "FAIL", "llm_judge (MOCK)", "empty reply", reply, extra=meta)
        reply_lower = reply.lower()
        if any(p in reply_lower for p in self._BAIL_PHRASES):
            return ScoreResult(example.id, "FAIL", "llm_judge (MOCK)",
                                "reply looks like a bail-out", reply, extra=meta)
        return ScoreResult(example.id, "PASS", "llm_judge (MOCK)",
                            "non-empty, non-bail-out reply — MOCK judge, not a real evaluation", reply, extra=meta)
