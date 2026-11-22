"""services/http_client.py: one retry policy for every public geocoding call.

The point of the module is that a timeout or a 5xx is retried the same way everywhere, and
that a 4xx is not (retrying a malformed address just wastes the caller's time). These tests
use a recording session, so they assert on attempts rather than on wall-clock behaviour.
"""

from __future__ import annotations

import requests
import pytest

from services import http_client


class FakeResponse:
    def __init__(self, status_code: int, text: str = "{}"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {}


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


@pytest.fixture()
def fake(monkeypatch):
    holder = {}

    def install(outcomes):
        session = FakeSession(outcomes)
        holder["session"] = session
        monkeypatch.setattr(http_client, "session", lambda: session)
        monkeypatch.setattr(http_client.time, "sleep", lambda _s: None)
        return session

    return install


def test_retries_a_server_error_then_succeeds(fake):
    session = fake([FakeResponse(502), FakeResponse(503), FakeResponse(200)])
    response = http_client.get("https://example.test/x", retries=2)
    assert response.status_code == 200
    assert len(session.calls) == 3


def test_gives_up_after_the_retry_budget_and_raises(fake):
    session = fake([FakeResponse(500), FakeResponse(500)])
    with pytest.raises(requests.HTTPError):
        http_client.get("https://example.test/x", retries=1)
    assert len(session.calls) == 2


def test_does_not_retry_a_client_error(fake):
    session = fake([FakeResponse(400)])
    with pytest.raises(requests.HTTPError):
        http_client.get("https://example.test/x", retries=2)
    assert len(session.calls) == 1


def test_retries_a_timeout_and_reraises_it_when_exhausted(fake):
    session = fake([requests.Timeout("slow"), requests.Timeout("slow")])
    with pytest.raises(requests.Timeout):
        http_client.get("https://example.test/x", retries=1)
    assert len(session.calls) == 2


def test_default_timeout_is_applied_and_can_be_overridden(fake):
    session = fake([FakeResponse(200), FakeResponse(200)])
    http_client.get("https://example.test/a")
    http_client.post("https://example.test/b", timeout=180, retries=0)
    assert session.calls[0][2]["timeout"] == http_client.DEFAULT_TIMEOUT
    assert session.calls[1][2]["timeout"] == 180
    assert session.calls[1][0] == "POST"
