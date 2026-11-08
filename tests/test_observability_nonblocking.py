"""Tracing must never slow a request: with Phoenix enabled and NO collector listening, spans must cost ~nothing.

Found in the field: the exporter was registered with batch=False, so every span was exported synchronously inside
span.end() and, with the collector down, burned ~7 seconds of connection retries - minutes per chat turn, and a 5-minute
test run.  Run in a fresh interpreter because OpenTelemetry's provider is process-global.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("phoenix.otel")
ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import os, sys, time
sys.path.insert(0, os.environ["APP_ROOT"]); os.chdir(os.environ["APP_ROOT"])
import observability as obs
obs.initialize_observability()
t0 = time.perf_counter()
for i in range(20):
    with obs.trace_span("probe", attributes={"i": i}):
        pass
print(f"ELAPSED {time.perf_counter() - t0:.3f}", flush=True)
os._exit(0)
"""


def run_probe(**env):
    full = dict(os.environ, APP_ROOT=str(ROOT), PHOENIX_ENABLED="true",
                PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:1/v1/traces", **env)          # nothing listens on port 1
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=full, timeout=120)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("ELAPSED")), None)
    assert line, out.stdout + out.stderr[-500:]
    return float(line.split()[1])


def test_twenty_spans_with_the_collector_down_cost_almost_nothing():
    assert run_probe() < 1.0
