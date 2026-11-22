"""
Playwright end-to-end browser verification for the LLM Thinking Process panel:

1. The Thinking Process side panel exists and toggles open/closed.
2. Sending a General Chat question opens the panel automatically and it starts at "● Starting".
3. The panel advances through the agent's own plan in real time and ends on "✓ result".
4. The final answer is still rendered in the chat as usual.
5. The finished step list stays in the panel after the turn completes.
6. Saves screenshots/15-llm-thinking-progress.png.

Run it directly (a local llama-server and loaded database are required, like tests/test_playwright_transit.py):

    python tests/test_playwright_progress.py
"""

import os
import sys
import threading
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):  # the panel's ●/✓/✗ bullets are not in the Windows console code page
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

# The agent's declared plan for General Chat (agents/general_agent.PROGRESS_STEPS).
EXPECTED_PLAN = [
    "Query Planning",
    "Data Model Retrieval",
    "Code Generation",
    "Code Execution",
    "Answer Synthesis",
]


def wait_for_server(url: str, timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/api/agent/status", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def ensure_server():
    if wait_for_server(BASE_URL, timeout=1.0):
        return None

    import uvicorn
    from main import app

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_for_server(BASE_URL, timeout=25.0):
        raise RuntimeError("Failed to start test server on port 8000")
    return server


def run_llm_thinking_process_ui_verification():
    from playwright.sync_api import sync_playwright

    ensure_server()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 850})
        page = context.new_page()
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print("\n1. Navigating to application...", flush=True)
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # The panel and its toggle exist.
        page.wait_for_selector("#thinking-panel", state="attached", timeout=10000)
        toggle_btn = page.locator("#btn-toggle-thinking")
        assert toggle_btn.is_visible()

        print("2. Testing Thinking panel toggle button...", flush=True)
        toggle_btn.click()
        page.wait_for_selector("#thinking-panel.open", timeout=5000)
        page.locator("#btn-close-thinking").click()
        page.wait_for_selector("#thinking-panel:not(.open)", timeout=5000)

        print("3. Opening General Chat...", flush=True)
        page.locator("#btn-general-chat").click()
        page.wait_for_selector("#tab-general-chat", state="visible", timeout=10000)

        prompt = "Which 3 cities have the highest median house prices in the database?"
        page.locator("#general-chat-input").fill(prompt)
        print(f"4. Sending query: '{prompt}'...", flush=True)
        page.locator("#general-chat-send").click()

        # The panel opens by itself and starts at "● Starting" with the agent's plan already visible.
        page.wait_for_selector("#thinking-panel.open", timeout=5000)
        print("PASS: Thinking panel opened automatically on query submission.", flush=True)
        page.wait_for_function(
            "() => { const b = document.querySelector('#thinking-status-badge');"
            " return b && b.textContent.trim().toLowerCase() === 'thinking...'; }",
            timeout=5000,
        )
        page.wait_for_function(
            "() => { const el = document.querySelector('#thinking-progress-list');"
            " return el && /Starting/.test(el.innerText); }",
            timeout=5000,
        )
        running_bullet = page.locator("#thinking-progress-list .thinking-step").first.inner_text()
        assert running_bullet.startswith("●"), f"Expected the first step to be running, got {running_bullet!r}"
        print("PASS: Panel starts at '● Starting'.", flush=True)

        page.wait_for_selector("#thinking-progress-list .thinking-step", timeout=5000)
        plan_text = page.locator("#thinking-progress-list").inner_text()
        for step_title in EXPECTED_PLAN + ["result"]:
            assert step_title in plan_text, f"Panel should show '{step_title}' in its plan; got:\n{plan_text}"
        print("PASS: Panel lists the agent's full step plan as pending.", flush=True)

        # Real-time movement: a finished step and the running step coexist, and "Started" has been renamed.
        print("5. Observing live step transitions...", flush=True)
        page.wait_for_function(
            "() => document.querySelectorAll('#thinking-progress-list .thinking-step.step-completed').length >= 2"
            " && document.querySelectorAll('#thinking-progress-list .thinking-step.step-running').length === 1",
            timeout=120000,
        )
        live_steps = page.locator("#thinking-progress-list .thinking-step")
        live_texts = [live_steps.nth(i).inner_text().replace("\n", " ").strip() for i in range(live_steps.count())]
        running = [t for t in live_texts if t.startswith("●")]
        print("Steps mid-turn:", live_texts, flush=True)
        assert live_texts[0].startswith("✓") and "Started" in live_texts[0], (
            f"Expected '✓ Started' mid-turn: {live_texts}"
        )
        assert len(running) == 1 and not running[0].endswith("result"), (
            f"Expected exactly one non-result running step: {live_texts}"
        )
        assert not any(t.endswith("Starting") for t in live_texts), (
            f"'Starting' should have become 'Started': {live_texts}"
        )
        print("PASS: Panel advanced step by step while the turn was running.", flush=True)

        print("6. Waiting for the turn to finish (llama-server, up to 180s)...", flush=True)
        page.wait_for_selector("#general-chat-send:not([disabled])", timeout=180000)

        # Every step finished, "Started" was renamed, and "result" is last.
        page.wait_for_selector("#thinking-status-badge.completed", timeout=10000)
        badge_text = page.locator("#thinking-status-badge").inner_text().strip()
        assert badge_text.lower() == "completed", f"Unexpected badge text: {badge_text!r}"

        steps_locator = page.locator("#thinking-progress-list .thinking-step")
        count = steps_locator.count()
        step_texts = [steps_locator.nth(i).inner_text().strip() for i in range(count)]
        print("Final thinking steps:")
        for text in step_texts:
            print("  ", text)

        assert step_texts[0].startswith("✓") and "Started" in step_texts[0], "First step should be '✓ Started'"
        assert step_texts[-1].startswith("✓") and step_texts[-1].endswith("result"), "Last step should be '✓ result'"
        for title in EXPECTED_PLAN:
            assert any(t.startswith("✓") and title in t for t in step_texts), f"Missing completed step '{title}'"
        # Regression: the panel must not keep stale steps that never ran.
        assert not any("Data Retrieval" in t for t in step_texts), f"Stale step in panel: {step_texts}"
        # A retry adds its own "Code Generation (Attempt 2)" entry, so only a floor is asserted.
        assert len(step_texts) >= len(EXPECTED_PLAN) + 2, f"Unexpected step count: {step_texts}"
        print("PASS: Every step advanced to '✓' and the panel ends on '✓ result'.", flush=True)

        # The answer itself is still shown as usual.
        assistant_bubbles = page.locator("#general-chat-messages .msg.assistant .bubble")
        reply_text = assistant_bubbles.last.inner_text().strip()
        print(f"\nFinal assistant response ({len(reply_text)} chars):\n{reply_text[:300]}...\n", flush=True)
        assert len(reply_text) > 15, "Assistant response was empty or too short"
        print("PASS: Final result rendered in the chat as usual.", flush=True)

        # The last state persists until the next query.
        page.wait_for_timeout(1500)
        assert page.locator("#thinking-panel").evaluate("el => el.classList.contains('open')")
        assert steps_locator.count() == count
        assert all(
            page.locator("#thinking-progress-list .thinking-step").nth(i).inner_text().strip() == step_texts[i]
            for i in range(count)
        )
        print("PASS: Thinking process state persisted after the result appeared.", flush=True)

        os.makedirs(os.path.join(REPO_ROOT, "screenshots"), exist_ok=True)
        screenshot_file = os.path.join(REPO_ROOT, "screenshots", "15-llm-thinking-progress.png")
        page.screenshot(path=screenshot_file, full_page=False)
        print(f"PASS: Captured screenshot to {screenshot_file} ({os.path.getsize(screenshot_file)} bytes)", flush=True)

        if console_errors:
            print("Browser console errors:", console_errors, flush=True)
        assert not console_errors, f"Browser console reported errors: {console_errors}"

        browser.close()

    print("\nAll thinking-process UI checks passed.", flush=True)


if __name__ == "__main__":
    run_llm_thinking_process_ui_verification()
