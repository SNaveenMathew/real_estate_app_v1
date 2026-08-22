#!/usr/bin/env python3
"""
run_eval.py — agent evaluation pipeline.

Runs every example in eval/golden_set.py against the REAL agents
(agents/general_agent.py, agents/house_agent.py) on a dedicated evaluation
database (eval/fixtures.py), scores each one, and writes a report.

Structured examples are scored by exact assert-equal logic (eval/scoring.py):
ordered sequence comparison when order matters, set comparison otherwise.
Free-text examples are scored by an LLM judge (eval/judge.py) against a
rubric. See those modules' docstrings for how each works.

Usage:
    python run_eval.py                                  # full run
    python run_eval.py --ids cities_with_houses,total_house_count
    python run_eval.py --tags nri,sold_homes
    python run_eval.py --skip-fixture-build             # reuse existing fixture DB
    python run_eval.py --list                           # list examples, run nothing
    python run_eval.py --skip-house-agent                # evaluate General Chat examples only
    python run_eval.py --mock                           # smoke-test the harness only —
                                                          # no model server needed, NOT a
                                                          # real evaluation (see eval/judge.py)

Requires a reachable model server for a real run — set llama_server_base_url
(the agent under test) and judge_llama_server_base_url (the judge; defaults
to the same server/model, but an independent or stronger judge is stronger
evidence) in .env. Exits non-zero if anything failed or errored, so this is
safe to wire into CI.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from eval.fixtures import build_fixture
from eval.golden_set import GOLDEN_SET, GoldenExample
from eval.scoring import score_structured, ScoreResult
from eval.judge import LLMJudge, MockJudge


def run_example(example: GoldenExample, judge, mock: bool):
    """Get the agent's reply, then dispatch to the right scorer."""
    trace_id = None
    trace_url = None
    if not mock:
        try:
            from observability import initialize_observability, clear_local_trace_capture
            initialize_observability()
            clear_local_trace_capture()
        except Exception:
            pass

    if mock:
        from eval.mock_agent import mock_reply
        try:
            reply = mock_reply(example)
        except Exception as e:
            return ScoreResult(example.id, "ERROR", "n/a", f"mock agent failed: {e}",
                                extra={"traceback": traceback.format_exc()})
    else:
        try:
            if example.agent == "general":
                from agents.general_agent import run_general_chat
                reply, _, _, metadata = run_general_chat(example.question, history=[], include_metadata=True)
                trace_id = metadata.get("trace_id") if metadata else None
                trace_url = metadata.get("trace_url") if metadata else None
            else:
                from agents.house_agent import run_house_chat
                reply, _ = run_house_chat(example.house_id, example.question, history=[])
        except Exception as e:
            trace_id = None
            try:
                from observability import latest_local_trace_id
                trace_id = latest_local_trace_id()
            except Exception:
                pass
            extra = {"traceback": traceback.format_exc()}
            if trace_id:
                extra["trace_id"] = trace_id
            return ScoreResult(example.id, "ERROR", "n/a", f"agent call failed: {e}", extra=extra)

    if example.answer_type == "structured":
        result = score_structured(example, reply)
    else:
        result = judge.judge(example, reply)
    if trace_id:
        result.extra["trace_id"] = trace_id
        result.extra["trace_url"] = trace_url
    return result


def _filter_examples(args) -> list[GoldenExample]:
    examples = list(GOLDEN_SET)
    if args.ids:
        wanted = set(args.ids.split(","))
        examples = [e for e in examples if e.id in wanted]
    if args.tags:
        wanted_tags = set(args.tags.split(","))
        examples = [e for e in examples if wanted_tags & set(e.tags)]
    if args.skip_house_agent:
        examples = [e for e in examples if e.agent != "house"]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run the agent evaluation pipeline")
    parser.add_argument("--ids", help="comma-separated example ids to run (default: all)")
    parser.add_argument("--tags", help="comma-separated tags to filter by (default: all)")
    parser.add_argument("--skip-fixture-build", action="store_true",
                         help="reuse an already-built fixture DB instead of rebuilding it")
    parser.add_argument("--list", action="store_true", help="list matching examples and exit")
    parser.add_argument("--skip-house-agent", action="store_true",
                        help="skip examples assigned to the house agent; evaluate General Chat only")
    parser.add_argument("--mock", action="store_true",
                         help="smoke-test the harness with a scripted stub reply and a "
                              "keyword-based judge — no model server required. "
                              "NOT a real evaluation of the agent.")
    args = parser.parse_args()

    examples = _filter_examples(args)
    if not examples:
        print("No golden examples matched the given filters.")
        sys.exit(1)

    if args.list:
        for e in examples:
            print(f"  {e.id:<40s} [{e.agent}] [{e.answer_type}] tags={list(e.tags)}")
        return

    if not args.skip_fixture_build:
        print(f"Building evaluation fixture at {settings.eval_fixture_duckdb_path} ...")
        build_fixture(settings.eval_fixture_duckdb_path)
    settings.duckdb_path = settings.eval_fixture_duckdb_path

    judge = MockJudge() if args.mock else LLMJudge()
    if args.mock:
        print("*** --mock: smoke-testing the harness only. This is NOT a real evaluation. ***")

    results: list[tuple[GoldenExample, ScoreResult]] = []
    print(f"\nRunning {len(examples)} example(s){' [MOCK]' if args.mock else ''}...\n")
    for ex in examples:
        t0 = time.time()
        result = run_example(ex, judge, args.mock)
        if not args.mock:
            try:
                from observability import export_trace
                trace_path = settings.eval_traces_dir / f"{ex.id}.json"
                trace_id = result.extra.get("trace_id")
                if trace_id:
                    export_trace(trace_id, trace_path, metadata={
                        "eval_id": ex.id,
                        "agent": ex.agent,
                        "answer_type": ex.answer_type,
                        "tags": list(ex.tags),
                        "question": ex.question,
                        "verdict": result.verdict,
                        "method": result.method,
                        "reason": result.reason,
                        "reply": result.reply,
                        "trace_url": result.extra.get("trace_url"),
                    })
                else:
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(json.dumps({
                        "trace_id": None,
                        "span_count": 0,
                        "metadata": {
                            "eval_id": ex.id, "agent": ex.agent,
                            "answer_type": ex.answer_type, "tags": list(ex.tags),
                            "question": ex.question, "verdict": result.verdict,
                            "method": result.method, "reason": result.reason,
                            "reply": result.reply,
                            "note": "No OpenTelemetry trace was emitted by this agent path."
                        },
                        "spans": []
                    }, indent=2, ensure_ascii=False), encoding="utf-8")
                result.extra["trace_file"] = str(trace_path)
            except Exception as exc:
                result.extra["trace_export_error"] = str(exc)
        dt = time.time() - t0
        icon = {"PASS": "\u2713", "FAIL": "\u2717", "ERROR": "!"}[result.verdict]
        print(f"  [{icon}] {ex.id:<42s} {result.verdict:<6s} ({result.method}, {dt:.1f}s)")
        if result.verdict != "PASS":
            print(f"        {result.reason}")
            if result.verdict == "ERROR" and result.extra.get("traceback"):
                print("        (full traceback in the report)")
        results.append((ex, result))

    _write_report(results, mock=args.mock)
    _print_summary(results)

    if any(r.verdict != "PASS" for _, r in results):
        sys.exit(1)


def _print_summary(results: list[tuple[GoldenExample, ScoreResult]]):
    n = len(results)
    passed = sum(1 for _, r in results if r.verdict == "PASS")
    failed = sum(1 for _, r in results if r.verdict == "FAIL")
    errored = sum(1 for _, r in results if r.verdict == "ERROR")
    print(f"\n{'=' * 60}")
    print(f"{passed}/{n} passed, {failed} failed, {errored} errored")

    for label, method_prefix in [("assert-equal (structured)", "assert_equal"),
                                  ("LLM judge (free-text)", "llm_judge")]:
        subset = [(e, r) for e, r in results if r.method.startswith(method_prefix)]
        if subset:
            p = sum(1 for _, r in subset if r.verdict == "PASS")
            print(f"  {label}: {p}/{len(subset)}")

    by_tag: dict[str, list[int]] = {}
    for ex, r in results:
        for tag in ex.tags:
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][1] += 1
            if r.verdict == "PASS":
                by_tag[tag][0] += 1
    if by_tag:
        print("\nBy dataset:")
        for tag, (p, t) in sorted(by_tag.items()):
            print(f"  {tag:<12s} {p}/{t}")


def _write_report(results: list[tuple[GoldenExample, ScoreResult]], mock: bool):
    settings.eval_reports_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = settings.eval_reports_dir / f"report-{ts}.json"
    md_path = settings.eval_reports_dir / f"report-{ts}.md"

    payload = {
        "timestamp": ts,
        "mock": mock,
        "results": [
            {
                "id": ex.id, "agent": ex.agent, "tags": list(ex.tags),
                "answer_type": ex.answer_type, "question": ex.question,
                "verdict": r.verdict, "method": r.method, "reason": r.reason,
                "reply": r.reply, "traceback": r.extra.get("traceback"),
                "trace_id": r.extra.get("trace_id"), "trace_url": r.extra.get("trace_url"),
                "trace_file": r.extra.get("trace_file"), "trace_export_error": r.extra.get("trace_export_error"),
            }
            for ex, r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    n = len(results)
    passed = sum(1 for _, r in results if r.verdict == "PASS")
    lines = [f"# Evaluation report — {ts}", ""]
    if mock:
        lines.append("**MOCK RUN — smoke-test only, not a real evaluation.**\n")
    lines.append(f"**{passed}/{n} passed**\n")
    lines.append("| id | agent | type | verdict | method |")
    lines.append("|---|---|---|---|---|")
    for ex, r in results:
        lines.append(f"| {ex.id} | {ex.agent} | {ex.answer_type} | {r.verdict} | {r.method} |")
    lines.append("\n## Details")
    for ex, r in results:
        reply_snippet = (r.reply or "")[:500]
        lines.append(f"\n### {ex.id} — {r.verdict}")
        lines.append(f"- **question:** {ex.question}")
        lines.append(f"- **reason:** {r.reason}")
        lines.append(f"- **reply:** {reply_snippet}")
        if r.extra.get("trace_file"):
            lines.append(f"- **trace file:** `{r.extra['trace_file']}`")
        if r.extra.get("trace_id"):
            lines.append(f"- **trace id:** `{r.extra['trace_id']}`")
        if r.extra.get("trace_export_error"):
            lines.append(f"- **trace export error:** `{r.extra['trace_export_error']}`")
        if r.verdict == "ERROR" and r.extra.get("traceback"):
            lines.append(f"- **traceback:**\n```\n{r.extra['traceback']}```")
    md_path.write_text("\n".join(lines))

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
