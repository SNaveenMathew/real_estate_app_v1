"""db/duckdb_store.py: per-thread connections, bounded statements, and the tool layer's handling of both."""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("dbconc"))
    yield f
    census_fixture.deactivate(f)


def store():
    import db.duckdb_store as s
    return s


def test_each_thread_gets_its_own_cursor_and_keeps_it(fx):
    s = store()
    mine = s.get_conn()
    assert s.get_conn() is mine
    seen = []
    t = threading.Thread(target=lambda: seen.extend([s.get_conn(), s.get_conn()]))
    t.start(); t.join()
    assert seen[0] is seen[1] and seen[0] is not mine


def test_a_cursor_sees_writes_committed_by_another_thread(fx):
    s = store()
    s.get_conn().execute("CREATE TABLE IF NOT EXISTS scratch_visibility (k INTEGER)")
    t = threading.Thread(target=lambda: s.get_conn().execute("INSERT INTO scratch_visibility VALUES (1), (2)"))
    t.start(); t.join()
    assert s.query("SELECT COUNT(*) AS n FROM scratch_visibility").iloc[0, 0] == 2


def test_query_limited_returns_the_same_frame_as_query(fx):
    s = store()
    for sql in ("SELECT tract_fips, population, name FROM census_tracts ORDER BY tract_fips",
                "SELECT city, AVG(price) AS p, COUNT(*) AS n, MAX(created_at) AS t FROM houses GROUP BY city ORDER BY city",
                "SELECT i, i::DOUBLE AS d, NULL AS nothing, DATE '2024-01-01' + i::INTEGER AS day FROM range(10) t(i)"):
        frame, truncated = s.query_limited(sql)
        pd.testing.assert_frame_equal(frame, s.query(sql))
        assert truncated is False


def test_the_row_cap_is_exact(fx):
    s = store()
    for rows, cap, expect_trunc in ((100_000, 5_000, True), (4_096, 4_096, False), (4_097, 4_096, True), (10, 10, False), (0, 5, False)):
        frame, truncated = s.query_limited(f"SELECT i FROM range({rows}) t(i)", max_rows=cap)
        assert len(frame) == min(rows, cap) and truncated is expect_trunc, (rows, cap)


def test_a_runaway_statement_is_cancelled_and_the_cursor_survives(fx):
    s = store()
    t0 = time.perf_counter()
    with pytest.raises(s.SqlTimeout):
        s.query_limited("SELECT COUNT(*) FROM range(1000000000) a, range(1000000000) b", timeout_s=0.7)
    assert time.perf_counter() - t0 < 5
    assert s.query("SELECT 41 + 1 AS x").iloc[0, 0] == 42


def test_a_fast_statement_is_not_disturbed_by_the_watchdog(fx):
    s = store()
    for _ in range(50):
        assert s.query_limited("SELECT 1 AS x", timeout_s=30)[0].iloc[0, 0] == 1
    time.sleep(0.05)
    assert threading.active_count() < 60            # the timers are cancelled, not left running


def test_concurrent_readers_and_writers_never_corrupt_each_other(fx):
    """12 threads x 25 rounds of mixed reads and writes, started together.  Every read must return the right frame."""
    s = store()
    s.get_conn().execute("CREATE TABLE IF NOT EXISTS scratch_stress (thread_id INTEGER, round INTEGER)")
    expected_tracts = int(s.query("SELECT COUNT(*) AS n FROM census_tracts").iloc[0, 0])
    expected_houses = int(s.query("SELECT COUNT(*) AS n FROM houses").iloc[0, 0])
    barrier, errors, reads = threading.Barrier(12), [], []

    def work(tid):
        try:
            barrier.wait()
            for r in range(25):
                tracts = s.query("SELECT COUNT(*) AS n FROM census_tracts")
                frame, _ = s.query_limited("SELECT COUNT(*) AS n FROM houses")
                assert tracts is not None and frame is not None
                reads.append((int(tracts.iloc[0, 0]), int(frame.iloc[0, 0])))
                s.get_conn().execute("INSERT INTO scratch_stress VALUES (?, ?)", [tid, r])
        except Exception as exc:                          # pragma: no cover - the assertion below reports it
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors, errors[:3]
    assert set(reads) == {(expected_tracts, expected_houses)} and len(reads) == 300
    assert s.query("SELECT COUNT(*) AS n FROM scratch_stress").iloc[0, 0] == 300


def test_the_old_shared_connection_pattern_really_does_corrupt_results(fx):
    """Teeth: the same workload on ONE shared connection (what get_conn() used to return) fails.  If this ever stops
    failing, DuckDB made connections thread-safe and the per-thread cursors can be reconsidered."""
    s = store()
    shared = s.get_conn().cursor()                       # one connection object used by every thread
    barrier, failures = threading.Barrier(12), []

    def work():
        barrier.wait()
        for _ in range(60):
            try:
                df = shared.execute("SELECT COUNT(*) AS n FROM census_tracts").df()
                if df is None or int(df.iloc[0, 0]) <= 0:
                    failures.append("wrong")
            except Exception as exc:
                failures.append(type(exc).__name__)

    threads = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert failures, "expected the shared-connection pattern to fail under concurrency on this DuckDB version"


def test_close_releases_the_cursors_and_the_next_call_reopens(fx):
    s = store()
    before = s.connection_generation()
    old = s.get_conn()
    s.close()
    assert not s.is_connected() and s._cursors == []
    fresh = s.get_conn()
    assert s.connection_generation() == before + 1 and fresh is not old
    assert s.query("SELECT COUNT(*) AS n FROM census_tracts").iloc[0, 0] > 0


class StubSql:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        return types.SimpleNamespace(content=self.replies.pop(0) if self.replies else "")


def test_a_timeout_is_a_status_and_is_never_retried(fx, monkeypatch):
    import agents.tools as tools
    from agents import answer_status
    s = store()
    stub = StubSql(["SELECT COUNT(houses.house_id) FROM houses"] * 3)
    monkeypatch.setattr(tools, "_agent", stub)

    def slow(*a, **k):
        raise s.SqlTimeout("The statement ran longer than 30 seconds and was cancelled.")

    monkeypatch.setattr(s, "query_limited", slow)
    out = tools.query_database.invoke({"request": "How many houses are there?"})
    body = out.split("[RESULT]\n", 1)[1]
    assert answer_status.status_code(body) == "limit_exceeded" and "too broad" in body
    assert stub.calls == 1                              # re-running the same slow plan would only repeat the wait


def test_truncation_is_reported_to_the_answer_model(fx, monkeypatch):
    import agents.tools as tools
    from config import settings
    monkeypatch.setattr(settings, "sql_max_rows", 4)
    monkeypatch.setattr(tools, "_agent", StubSql(["SELECT census_tracts.tract_fips, census_tracts.population FROM census_tracts"]))
    body = tools.query_database.invoke({"request": "list every census tract population"}).split("[RESULT]\n", 1)[1]
    assert "more than 4 rows exist" in body and "add a filter" in body
