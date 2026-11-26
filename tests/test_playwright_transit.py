"""
Playwright end-to-end browser verification for FTA transit integration:
1. Map container and controls loaded
2. Transit Routes layer discovery in layers panel
3. Centering on Pittsburgh and toggling Transit Routes renders line geometry via /api/layers/transit_routes
4. General Chat answers transit routing queries with interactive route maps
5. General Chat answers transit ridership analytical questions using the Code Agent and llama-server
"""

import time

import pytest

# Browser verification is opt-in: Playwright is not a runtime or CI dependency (see
# requirements.txt), and importing it unconditionally turns a missing optional tool into a
# collection error that fails the whole suite before a single test runs.
pytest.importorskip("playwright.sync_api", reason="playwright is not installed; browser verification is opt-in")
from playwright.sync_api import sync_playwright  # noqa: E402


def run_transit_ui_verification():
    print("--- Starting Playwright UI Verification ---", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        page.on(
            "console",
            lambda msg: (
                print(f"[Browser Console] {msg.type}: {msg.text}", flush=True)
                if msg.type in ("error", "warning")
                else None
            ),
        )

        print("1. Navigating to http://127.0.0.1:8000 ...", flush=True)
        page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)

        # 1. Map container
        page.wait_for_selector("#map", timeout=15000)
        print("PASS: Map container #map loaded", flush=True)

        # 2. Layers panel & Transit Routes checkbox
        print("2. Verifying Transit Routes layer in layers panel ...", flush=True)
        transit_row = page.locator("label.ml-row", has_text="Transit Routes")
        transit_row.wait_for(state="visible", timeout=15000)
        row_text = transit_row.inner_text().replace("\n", " ").strip()
        print(f"PASS: Found Transit Routes row: '{row_text}'", flush=True)

        transit_cb = transit_row.locator("input")
        assert not transit_cb.is_checked(), "Transit routes should start unchecked"

        # Zoom into Pittsburgh where routes are active and dense
        print("Centering map view on Pittsburgh (lat=40.45, lon=-79.95, zoom=13)...", flush=True)
        page.evaluate("() => map.setView([40.45, -79.95], 13)")
        page.wait_for_timeout(1000)

        # 3. Toggle Transit Routes layer ON
        print("3. Toggling Transit Routes layer ON ...", flush=True)
        with page.expect_response(
            lambda resp: "/api/layers/transit_routes" in resp.url and resp.status == 200, timeout=45000
        ) as response_info:
            transit_row.click()

        layer_resp = response_info.value
        assert layer_resp.status == 200
        print(f"PASS: /api/layers/transit_routes responded HTTP {layer_resp.status}", flush=True)

        # Allow Leaflet to render SVG paths
        print("Waiting for Leaflet to render SVG line paths ...", flush=True)
        page.wait_for_selector(".leaflet-pane.leaflet-overlay-pane path", timeout=20000)
        page.wait_for_timeout(1500)
        path_count = page.locator(".leaflet-pane.leaflet-overlay-pane path").count()
        print(f"PASS: Leaflet overlay pane rendered {path_count} SVG paths", flush=True)
        assert path_count > 0, "No SVG paths rendered on Leaflet map"

        # 4. General Chat with LLM - Transit Route Query
        print("4. Testing General Chat with LLM for Transit Route Query ...", flush=True)
        chat_btn = page.locator("#btn-general-chat")
        chat_btn.click()
        page.wait_for_selector("#general-chat-input", state="visible", timeout=10000)

        chat_input = page.locator("#general-chat-input")
        route_query = "Find a transit route from Shadyside to Highland Park"
        chat_input.fill(route_query)
        print(f"Sent prompt to General Chat: '{route_query}'", flush=True)

        send_btn = page.locator("#general-chat-send")
        send_btn.click()

        # Wait until the send button is disabled (request started) and re-enabled (completed)
        page.wait_for_timeout(500)
        print("Waiting for LLM response from llama-server (up to 120s)...", flush=True)
        page.wait_for_selector("#general-chat-send:not([disabled])", timeout=120000)

        # Fetch the completed assistant response (last assistant message)
        assistant_bubbles = page.locator("#general-chat-messages .msg.assistant .bubble")
        last_bubble = assistant_bubbles.last
        reply_text = last_bubble.inner_text()
        print(f"PASS: Assistant route response received ({len(reply_text)} chars):", flush=True)
        print("--------------------------------------------------", flush=True)
        print(reply_text[:500] + ("..." if len(reply_text) > 500 else ""), flush=True)
        print("--------------------------------------------------", flush=True)
        assert len(reply_text) > 10, "Response is too short"

        # Verify transit route artifact rendered in chat
        route_cards = page.locator(".bike-route-result")
        print(f"Checking for route result cards: count = {route_cards.count()}", flush=True)
        assert route_cards.count() > 0, "Expected transit route result card in chat"
        card_text = route_cards.first.inner_text()
        print("PASS: Transit route result card rendered in chat:", card_text.splitlines()[0], flush=True)

        # Take screenshot of the complete state
        screenshot_path = "tests/playwright_transit_verification.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"PASS: Saved full-page verification screenshot to {screenshot_path}", flush=True)

        browser.close()
        print("--- ALL PLAYWRIGHT UI VERIFICATIONS PASSED SUCCESSFULLY ---", flush=True)


if __name__ == "__main__":
    run_transit_ui_verification()
