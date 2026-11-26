"""The judge retries once, then says INCONCLUSIVE instead of calling a good answer ERROR."""

import pytest

from agents import llm_resilience as lr
from eval.golden_set import GoldenExample
from eval.judge import LLMJudge


@pytest.fixture(autouse=True)
def _reset_breakers():
    lr.reset()
    yield
    lr.reset()


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Scripted:
    """Stands in for ChatOpenAI: returns/raises whatever the test queued."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Message(reply)


def _example() -> GoldenExample:
    return GoldenExample(
        id="judge_case",
        agent="general",
        question="How many houses?",
        answer_type="free_text",
        rubric="It should say six.",
    )


def _judge(replies) -> LLMJudge:
    judge = LLMJudge(base_url="http://127.0.0.1:9/v1", model="judge-model")
    judge._llm = _Scripted(replies)  # no network, no model server
    return judge


def test_a_valid_verdict_on_the_first_attempt_costs_one_call():
    judge = _judge(['{"verdict": "PASS", "reasoning": "says six"}'])
    result = judge.judge(_example(), "There are six houses.")
    assert result.verdict == "PASS"
    assert result.extra["judge_attempts"] == 1
    assert result.extra["judge_model"] == "judge-model"


def test_prose_instead_of_json_is_retried_and_then_succeeds():
    judge = _judge(
        [
            "I think this answer is probably fine, but I am not going to emit JSON.",
            '{"verdict": "PASS", "reasoning": "says six"}',
        ]
    )
    result = judge.judge(_example(), "There are six houses.")
    assert result.verdict == "PASS"
    assert result.extra["judge_attempts"] == 2
    assert judge._llm.calls == 2


def test_two_unusable_answers_are_inconclusive_not_error():
    """The answer was never judged, so it must not be recorded as a failure of the agent."""
    judge = _judge(["not json at all", "still not json"])
    result = judge.judge(_example(), "There are six houses.")
    assert result.verdict == "INCONCLUSIVE"
    assert result.verdict not in ("PASS", "FAIL", "ERROR")
    assert result.extra["judge_attempts"] == 2
    assert "parseable JSON" in result.reason


def test_a_transport_failure_is_retried_too():
    judge = _judge(
        [
            RuntimeError("connection refused"),
            '{"verdict": "FAIL", "reasoning": "it says five"}',
        ]
    )
    result = judge.judge(_example(), "There are five houses.")
    assert result.verdict == "FAIL"
    assert result.extra["judge_attempts"] == 2


def test_an_unrecognised_verdict_is_also_inconclusive():
    judge = _judge(['{"verdict": "MAYBE"}', '{"verdict": "MAYBE"}'])
    result = judge.judge(_example(), "There are six houses.")
    assert result.verdict == "INCONCLUSIVE"
    assert result.extra["inconclusive_kind"] == "inconclusive_schema"


def test_key_typo_vertext_is_rejected_and_retried():
    judge = _judge(['{"vertext": "FAIL", "reasoning": "typo"}', '{"verdict": "FAIL", "reasoning": "wrong count"}'])
    result = judge.judge(_example(), "There are five houses.")
    assert result.verdict == "FAIL"
    assert result.extra["judge_attempts"] == 2


def test_long_reply_is_truncated_within_judge_prompt_budget(monkeypatch):
    from config import settings

    seen_prompts = []

    class CapturingClient:
        def invoke(self, messages):
            seen_prompts.append(str(messages[-1].content))
            return _Message('{"verdict": "PASS", "reasoning": "ok"}')

    monkeypatch.setattr(settings, "judge_prompt_max_chars", 2000)
    judge = LLMJudge(base_url="http://127.0.0.1:9/v1", model="judge-model")
    judge._llm = CapturingClient()
    huge_reply = "START_MARKER " + ("x" * 20000) + " END_MARKER"
    result = judge.judge(_example(), huge_reply)
    assert result.verdict == "PASS"
    assert len(seen_prompts[0]) <= 2200
    assert "START_MARKER" in seen_prompts[0] and "END_MARKER" in seen_prompts[0]


def test_artifacts_are_included_in_judge_prompt():
    seen_prompts = []

    class CapturingClient:
        def invoke(self, messages):
            seen_prompts.append(str(messages[-1].content))
            return _Message('{"verdict": "PASS", "reasoning": "table present"}')

    judge = LLMJudge(base_url="http://127.0.0.1:9/v1", model="judge-model")
    judge._llm = CapturingClient()
    result = judge.judge(
        _example(),
        "See the table beside this answer.",
        artifacts=[{"kind": "table", "title": "Saved houses", "columns": ["house_id"], "rows": [{"house_id": "h1"}]}],
    )
    assert result.verdict == "PASS"
    assert "Saved houses" in seen_prompts[0] and "house_id" in seen_prompts[0]


def test_outlines_structured_client_builds_strict_json_schema_generator():
    from eval.judge import JudgeVerdict
    from services.guardrails import OutlinesStructuredClient, inline_json_schema

    flat = inline_json_schema(JudgeVerdict)
    assert set(flat["properties"]) == {"verdict", "reasoning"}
    client = OutlinesStructuredClient(
        base_url="http://127.0.0.1:9/v1",
        api_key="not-needed",
        model="Qwen3.5-4B",
        output_schema=JudgeVerdict,
        max_tokens=128,
        stop=["<end_of_turn>"],
    )
    assert client.max_tokens == 128 and client._generator is not None
