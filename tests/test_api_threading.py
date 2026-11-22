"""A slow request must not freeze every other request.

The measured failure this guards (audit §9.3): with every handler declared ``async def``
but calling a blocking DuckDB read, one map pan - a 1,971 ms crime-heat query - held the
event loop and pushed an unrelated 15.5 ms ``/api/config`` to 1,822 ms. Handlers that only
block are now plain ``def``, so FastAPI runs them in its worker pool and the per-thread
cursors in db/duckdb_store.py serve each one.

Both requests are started before either is awaited, and each records when its response
arrived. On a blocked loop the fast one cannot finish until the slow one is done; with the
work off the loop it finishes while the slow one is still running.
"""
import asyncio
import time

import httpx
import pytest


@pytest.fixture()
def app(reference_data, tmp_path, monkeypatch):
    from config import settings
    from services import dataset_onboarding as ob

    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(ob, "warm_vector_index", lambda: False)
    import main
    return main.app


def test_a_slow_db_handler_leaves_the_loop_free_for_other_requests(app, monkeypatch):
    import db.duckdb_store as store

    fast_reads = store.get_all_houses

    def slow_reads():
        time.sleep(0.4)                     # stands in for the measured 1.9 s crime-heat query
        return fast_reads()

    monkeypatch.setattr(store, "get_all_houses", slow_reads)

    async def scenario() -> tuple[float, float]:
        transport = httpx.ASGITransport(app=app)
        ready: dict[str, float] = {}
        started = time.perf_counter()

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async def fetch(name: str, url: str) -> int:
                response = await client.get(url)
                ready[name] = time.perf_counter() - started
                return response.status_code

            slow = asyncio.create_task(fetch("slow", "/api/houses"))
            fast = asyncio.create_task(fetch("fast", "/api/config"))
            assert await asyncio.gather(slow, fast) == [200, 200]

        return ready["fast"], ready["slow"]

    fast_done, slow_done = asyncio.run(scenario())
    assert slow_done >= 0.35, "the slow request was expected to still be working when the fast one finished"
    assert fast_done < 0.25, f"the fast request waited {fast_done:.2f}s behind the slow one: a handler is back on the loop"
