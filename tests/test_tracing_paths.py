"""The agent code paths that only run when tracing is ON, exercised with an in-memory tracer (no network, no Phoenix).

Every other test in this suite runs with tracing off (see census_fixture), because a test must not depend on - or wait for -
an external collector.  That left a blind spot: all the ``if span is not None`` code.  This module closes it, using the
real OpenTelemetry SDK with an in-memory exporter, so span attributes, outputs and error recording are really executed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

sdk = pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import census_fixture  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("tracing"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def spans(fx, monkeypatch):
    import observability
    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_tracer", provider.get_tracer("test"))
    import agents.tools as tools

    class Sql:
        def invoke(self, messages, **kw):
            return types.SimpleNamespace(content="SELECT COUNT(houses.house_id) FROM houses WHERE houses.city = 'Pittsburgh'")

    monkeypatch.setattr(tools, "_agent", Sql())
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    return exporter


def names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def ask(request):
    import agents.tools as tools
    return tools.query_database.invoke({"request": request})


def test_the_sql_path_records_its_spans_and_still_answers(spans):
    out = ask("How many houses are in Pittsburgh?")
    assert out.split("[RESULT]\n", 1)[1].split()[-1] == "2"
    assert {"sql_generation", "sql_validation", "sql_execution"} <= set(names(spans))
    execution = next(s for s in spans.get_finished_spans() if s.name == "sql_execution")
    assert execution.attributes["sql.row_count"] == 1 and execution.attributes["sql.truncated"] is False


def test_the_derived_measure_path_records_provider_and_measures(spans, fx):
    out = ask("Verify the land areas of Pittsburgh and Indianapolis")
    assert f"{fx.expected['Pittsburgh, PA Metro Area']['area_sq_mi']:.2f}" in out
    span = next(s for s in spans.get_finished_spans() if s.name == "derived_measures")
    assert span.attributes["derived.provider"] == "msa_geometry"


def test_a_contract_violation_and_its_repair_are_traced_without_error(spans):
    import agents.tools as tools
    tools._agent = types.SimpleNamespace(invoke=lambda m, **k: types.SimpleNamespace(
        content="SELECT COUNT(*) FROM houses, census_tracts WHERE houses.city = 'Pittsburgh'"))         # cross join: rejected, then compiled
    out = ask("How many houses are in Pittsburgh?")
    assert "houses.city = 'Pittsburgh'" in out and "NOT_ANSWERED" not in out
    assert "sql_validation" in names(spans)


def test_a_timeout_is_traced_and_reported(spans, monkeypatch):
    import db.duckdb_store as store
    monkeypatch.setattr(store, "query_limited", lambda *a, **k: (_ for _ in ()).throw(store.SqlTimeout("ran too long")))
    body = ask("How many houses are in Pittsburgh?").split("[RESULT]\n", 1)[1]
    assert body.startswith("NOT_ANSWERED[limit_exceeded]")
    assert "sql_execution" in names(spans)


def test_a_whole_orchestrated_turn_with_tracing_on(spans, fx, monkeypatch):
    import agents.general_agent as ga
    code = types.SimpleNamespace(invoke=lambda m, **k: types.SimpleNamespace(content=""))
    echo = types.SimpleNamespace(invoke=lambda m, **k: types.SimpleNamespace(content=str(m[0].content).split("EXECUTED EVIDENCE:", 1)[-1]))
    monkeypatch.setattr(ga, "_get_code_agent", lambda: code)
    monkeypatch.setattr(ga, "_get_response_agent", lambda: echo)
    reply = ga.run_general_chat("What is the population of Pittsburgh?")[0]
    assert "24000" in reply and len(names(spans)) >= 3
