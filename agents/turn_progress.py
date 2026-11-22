"""Real-time progress tracker for agent execution steps."""
from __future__ import annotations

import contextvars
import threading
import time
from typing import Any, Callable

_CURRENT_TRACKER: contextvars.ContextVar[TurnProgressTracker | None] = contextvars.ContextVar("turn_progress_tracker", default=None)


def get_current_tracker() -> TurnProgressTracker | None:
    return _CURRENT_TRACKER.get()


def set_current_tracker(tracker: TurnProgressTracker | None):
    return _CURRENT_TRACKER.set(tracker)


# Bullet symbols matching specification:
# ● (running / active)
# ✓ (completed)
# o (pending)
# ✗ (failed)
BULLET_RUNNING = "●"
BULLET_COMPLETED = "✓"
BULLET_PENDING = "o"
BULLET_FAILED = "✗"


class TurnProgressTracker:
    def __init__(self, turn_id: str, default_steps: list[tuple[str, str]] | None = None):
        self.turn_id = turn_id
        self.created_at = time.time()
        self._lock = threading.RLock()
        self._is_complete = False

        # Build initial step list
        # Contract:
        # - Starting step: "Starting" (running initially)
        # - Intermediate steps: pending initially, declared by the agent that will run them
        # - Result step: "result" (pending initially)
        # ``default_steps`` is the agent's own plan (agents/general_agent.PROGRESS_STEPS /
        # agents/house_agent.PROGRESS_STEPS), so the panel never shows steps that will not run.
        self.steps: list[dict[str, str]] = [
            {"id": "start", "title": "Starting", "status": "running"}
        ]
        for step_id, title in default_steps or []:
            self.steps.append({"id": step_id, "title": title, "status": "pending"})
        self.steps.append({"id": "result", "title": "result", "status": "pending"})

        self._listeners: list[Callable[[dict], None]] = []

    def add_listener(self, callback: Callable[[dict], None]):
        with self._lock:
            self._listeners.append(callback)
            snapshot = self.get_state()
        try:
            callback(snapshot)
        except Exception:
            pass

    def remove_listener(self, callback: Callable[[dict], None]):
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify(self):
        snapshot = self.get_state()
        listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(snapshot)
            except Exception:
                pass

    def start_step(self, step_id: str, title: str):
        """Begin or activate a step, marking previous in-progress steps completed."""
        with self._lock:
            if self._is_complete:
                return

            # When any non-start step starts, 'Starting' switches to 'Started' and completed
            start_step = next((s for s in self.steps if s["id"] == "start"), None)
            if start_step:
                start_step["status"] = "completed"
                start_step["title"] = "Started"

            # Check if step exists by id or title
            existing = next((s for s in self.steps if s["id"] == step_id or s["title"].lower() == title.lower()), None)
            if existing:
                existing["id"] = step_id
                existing["title"] = title
                existing["status"] = "running"
            else:
                # Insert right before pending steps or result
                insert_idx = len(self.steps) - 1
                for i, s in enumerate(self.steps):
                    if s["id"] == "result" or s["status"] == "pending":
                        insert_idx = i
                        break
                new_step = {"id": step_id, "title": title, "status": "running"}
                self.steps.insert(insert_idx, new_step)

            # Ensure any other running step prior to this one is marked completed
            for s in self.steps:
                if s["id"] not in (step_id, "result") and s["status"] == "running":
                    s["status"] = "completed"

        self._notify()

    def complete_step(self, step_id: str):
        with self._lock:
            for s in self.steps:
                if s["id"] == step_id:
                    s["status"] = "completed"
                    if s["id"] == "start":
                        s["title"] = "Started"
        self._notify()

    def set_result_running(self):
        with self._lock:
            for s in self.steps:
                if s["id"] != "result" and s["status"] != "failed":
                    s["status"] = "completed"
                    if s["id"] == "start":
                        s["title"] = "Started"
                elif s["id"] == "result":
                    s["status"] = "running"
        self._notify()

    def complete_all(self):
        with self._lock:
            for s in self.steps:
                s["status"] = "completed"
                if s["id"] == "start":
                    s["title"] = "Started"
            self._is_complete = True
        self._notify()

    def fail_current(self, err_msg: str = ""):
        with self._lock:
            for s in self.steps:
                if s["status"] == "running":
                    s["status"] = "failed"
                    if err_msg:
                        s["error"] = err_msg
            self._is_complete = True
        self._notify()

    def format_steps(self) -> list[str]:
        """Return formatted bullet strings conforming to the user specification."""
        formatted = []
        for s in self.steps:
            st = s.get("status", "pending")
            if st == "completed":
                bullet = BULLET_COMPLETED
                title = "Started" if s["id"] == "start" else s["title"]
            elif st == "running":
                bullet = BULLET_RUNNING
                title = "Starting" if s["id"] == "start" else s["title"]
            elif st == "failed":
                bullet = BULLET_FAILED
                title = s["title"]
            else:
                bullet = BULLET_PENDING
                title = s["title"]
            formatted.append(f"{bullet} {title}")
        return formatted

    def is_complete(self) -> bool:
        with self._lock:
            return self._is_complete

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turn_id": self.turn_id,
                "steps": [dict(s) for s in self.steps],
                "formatted": self.format_steps(),
                "is_complete": self._is_complete,
            }


_TRACKERS: dict[str, TurnProgressTracker] = {}
_TRACKERS_LOCK = threading.Lock()
_CLEANUP_AGE_SECONDS = 3600  # retain turns for 1 hour


def create_progress_tracker(turn_id: str, default_steps: list[tuple[str, str]] | None = None) -> TurnProgressTracker:
    with _TRACKERS_LOCK:
        now = time.time()
        expired = [tid for tid, t in _TRACKERS.items() if now - t.created_at > _CLEANUP_AGE_SECONDS]
        for tid in expired:
            _TRACKERS.pop(tid, None)

        tracker = TurnProgressTracker(turn_id, default_steps=default_steps)
        _TRACKERS[turn_id] = tracker
        return tracker


def get_progress_tracker(turn_id: str) -> TurnProgressTracker | None:
    with _TRACKERS_LOCK:
        return _TRACKERS.get(turn_id)


def get_or_create_progress_tracker(turn_id: str, default_steps: list[tuple[str, str]] | None = None) -> TurnProgressTracker:
    with _TRACKERS_LOCK:
        if turn_id in _TRACKERS:
            return _TRACKERS[turn_id]
    return create_progress_tracker(turn_id, default_steps=default_steps)
