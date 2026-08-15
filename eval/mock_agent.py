"""
eval/mock_agent.py

A scripted stand-in for the real agent, used only by `run_eval.py --mock`.
NOT a real agent — it never calls a model or touches the database. It exists
so the harness itself (fixture build -> scoring dispatch -> report writing)
can be smoke-tested with no model server reachable.

For structured examples it plainly restates `expected` (in the required
order, if order_matters) — this is a genuine self-check on the scorer: if
scoring.py had a bug that failed everything regardless of content, this
would catch it, the same way eval/tests/test_scoring.py's hand-crafted
PASS cases do. For free_text examples there's no principled way to
auto-satisfy an arbitrary rubric, so it returns an honest placeholder —
expect FAIL from a real judge, and a MOCK-labeled PASS from MockJudge
(which only checks the reply is non-empty and not a bail-out). That's fine:
--mock is about proving the harness runs, not about grading quality.
"""
from __future__ import annotations

from eval.golden_set import GoldenExample


def mock_reply(example: GoldenExample) -> str:
    if example.answer_type == "structured":
        if example.value_type == "string":
            vals = ", ".join(str(v) for v in example.expected)
            return f"Here you go: {vals}."
        prefix = "$" if example.value_type == "currency" else ""
        vals = ", ".join(f"{prefix}{v:,.0f}" if float(v).is_integer() else f"{prefix}{v:,}"
                          for v in example.expected)
        return f"The answer is {vals}."
    return "This is a placeholder reply for smoke-testing the harness (--mock)."
