"""The chat HTML sanitiser (static/sanitize.js) has real attacks in its test suite.

Those tests are JavaScript, so they live in tests/js/ and run under `node --test`; this
module makes pytest (and therefore CI) run them, and skips cleanly on a machine without Node
instead of pretending the sanitiser is covered.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "js" / "sanitize.test.mjs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed; the sanitiser suite cannot run")


def test_sanitiser_rejects_script_injection_vectors():
    proc = subprocess.run(
        [NODE, "--test", str(SUITE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fail 0" in proc.stdout


def test_chat_bubbles_go_through_the_sanitiser():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "sanitizeHtml(html)" in app_js
    assert 'innerHTML = `<div class="bubble">' not in app_js, "the markdown sink must not be raw innerHTML"
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert index.index("/static/sanitize.js") < index.index("/static/app.js")
    assert "Content-Security-Policy" in index
