"""The judge retries once, then says INCONCLUSIVE instead of calling a good answer ERROR."""

from eval.golden_set import GoldenExample
from eval.judge import LLMJudge


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
