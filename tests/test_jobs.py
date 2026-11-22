"""Background jobs (services/jobs.py): the contract the 202 endpoints rely on."""

import threading
import time

import pytest
from fastapi import HTTPException

from services import jobs


def _wait(job_id: str, timeout: float = 5.0):
    return jobs.wait(job_id, timeout=timeout).snapshot()


def test_a_job_reports_progress_and_its_result():
    seen = {}

    def work(job):
        job.progress("half way", done=1, total=2)
        seen.update(job.snapshot())
        return {"ok": True}

    snap = _wait(jobs.start("test-progress", "demo", work).id)
    assert snap["state"] == "done"
    assert snap["result"] == {"ok": True}
    assert seen["state"] == "running" and seen["done"] == 1 and seen["total"] == 2


def test_a_failing_job_records_its_error_instead_of_vanishing():
    def work(job):
        raise ValueError("the loader said no")

    snap = _wait(jobs.start("test-fail", "demo", work).id)
    assert snap["state"] == "error"
    assert "the loader said no" in snap["error"]
    assert snap["result"] is None


def test_an_http_exception_keeps_its_own_message():
    def work(job):
        raise HTTPException(status_code=422, detail="this upload wasn't applied")

    snap = _wait(jobs.start("test-http-error", "demo", work).id)
    assert snap["state"] == "error"
    assert snap["error"] == "this upload wasn't applied"  # no "HTTPException:" prefix


def test_one_job_per_kind_and_the_second_is_refused():
    release = threading.Event()

    def work(job):
        release.wait(2.0)

    first = jobs.start("test-busy", "first", work)
    try:
        with pytest.raises(jobs.JobBusy) as busy:
            jobs.start("test-busy", "second", work)
        assert busy.value.job.id == first.id
        # a different kind is independent
        other = jobs.start("test-busy-other", "other", lambda job: None)
        assert _wait(other.id)["state"] == "done"
    finally:
        release.set()
        _wait(first.id)


def test_a_finished_job_frees_its_kind():
    first = jobs.start("test-reuse", "first", lambda job: 1)
    _wait(first.id)
    second = jobs.start("test-reuse", "second", lambda job: 2)
    assert _wait(second.id)["state"] == "done"


def test_snapshots_are_pollable_and_listed():
    job = jobs.start("test-list", "listed", lambda j: "value")
    snap = _wait(job.id)
    assert snap["job_id"] == job.id and snap["kind"] == "test-list"
    listed = jobs.recent(50)
    assert any(entry["job_id"] == job.id for entry in listed)
    assert jobs.get("does-not-exist") is None


def test_wait_times_out_on_a_job_that_never_finishes():
    release = threading.Event()
    job = jobs.start("test-timeout", "slow", lambda j: release.wait(2.0))
    try:
        with pytest.raises(TimeoutError):
            jobs.wait(job.id, timeout=0.1)
    finally:
        release.set()
        jobs.wait(job.id, timeout=5)
