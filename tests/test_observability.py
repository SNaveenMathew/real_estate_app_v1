import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import observability


@pytest.mark.parametrize(
    ("start_chat", "end_chat", "start_args", "chat_name"),
    [
        (
            observability.start_general_chat,
            observability.end_general_chat,
            ("hello", "session", 0),
            "general_chat",
        ),
        (
            observability.start_house_chat,
            observability.end_house_chat,
            ("hello", "house-1", 0),
            "house_chat",
        ),
    ],
)
def test_chat_finalizer_exports_root_span_and_restores_context(
    monkeypatch, start_chat, end_chat, start_args, chat_name
):
    provider = TracerProvider()
    exporter = observability.EvaluationTraceExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_tracer_provider", provider)
    monkeypatch.setattr(observability, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(observability, "_local_trace_exporter", exporter)
    monkeypatch.setattr(observability, "initialize_observability", lambda: None)

    root_context, root_span, trace_id, _ = start_chat(*start_args)
    with observability.trace_span(f"{chat_name}.child"):
        pass

    assert observability.local_trace_span_count(trace_id) == 1

    end_chat(
        root_span,
        root_context=root_context,
        trace_id=trace_id,
        reply="done",
    )

    spans = exporter.snapshot(trace_id)
    assert len(spans) == 2
    assert {span.name for span in spans} == {chat_name, f"{chat_name}.child"}
    assert not trace_api.get_current_span().get_span_context().is_valid
    provider.shutdown()
