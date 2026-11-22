"""Polling endpoints for background jobs (services/jobs.py).

The heavy POSTs answer 202 with a ``job_id`` and the page polls here. Kept
separate from the routers that start the work so one endpoint serves every kind.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(limit: int = Query(default=20, ge=1, le=100)):
    return {"jobs": jobs.recent(limit)}


@router.get("/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have been evicted; jobs are in memory).")
    return job.snapshot()
