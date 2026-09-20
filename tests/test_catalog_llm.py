"""The catalog model router: structured output, validation, repair, fallback and offline degrade."""
import http.server
import json
import threading

import pytest

from config import settings
from services import catalog_llm as cl


class _Server:
    def __init__(self):
        self.script: list = []          # queued replies: str (content) or dict(status=..., body=...)
        self.seen: list[dict] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, {"data": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                item = outer.script.pop(0) if outer.script else outer.auto(body)
                if isinstance(item, dict) and "status" in item:
                    return self._send(item["status"], {"error": item.get("body", "bad request")})
                self._send(200, {"choices": [{"message": {"content": item}}], "usage": {"total_tokens": 10}})

        self.auto = lambda body: "{}"
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def llm(monkeypatch):
    srv = _Server()
    monkeypatch.setattr(settings, "catalog_draft_base_url", srv.url)
    monkeypatch.setattr(settings, "catalog_llm_enabled", True)
    monkeypatch.setattr(settings, "catalog_llm_timeout", 5.0)
    cl.reset_router()
    yield srv
    srv.close()
    cl.reset_router()


def _ds():
    def col(name, role, dtype="float"):
        return {"name": name, "source_name": name.upper(), "dtype": dtype, "role": role, "key_kind": None,
                "include": True, "stats": {"distinct": 50, "null_pct": 0.0}, "samples": ["1", "2"]}
    return {"source_filename": "x.csv", "title": "x", "row_count": 10, "columns": [
        col("geoid20", "key", "text"), col("walk_idx", "measure"), col("lead_pct", "measure"),
        col("cbsa_name", "dimension", "text"), col("d3b", "measure")]}


def _draft(names, **over):
    body = {"title": "Walkability", "description": "Walkability by block group.", "grain": "one row per block group",
            "domain": "mobility",
            "columns": [{"name": n, "description": f"About {n}.", "role": "measure", "unit": "index",
                         "synonyms": ["walkability index", "value", "walk"]} for n in names]}
    body.update(over)
    return json.dumps(body)


ALL = ["geoid20", "walk_idx", "lead_pct", "cbsa_name", "d3b"]


def test_json_schema_is_flat_and_constrains_roles():
    s = cl.json_schema(cl.DatasetDraft)
    assert "$defs" not in json.dumps(s) and "$ref" not in json.dumps(s)
    assert s["properties"]["columns"]["items"]["properties"]["role"]["enum"][0] == "key"


def test_extract_json_handles_think_blocks_and_fences():
    assert cl.extract_json('<think>hmm {x}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        cl.extract_json("no json here")


def test_draft_happy_path_is_constrained_bounded_and_sanitised(llm):
    llm.script = [_draft(ALL)]
    ds = _ds()
    res = cl.draft_dataset(ds)
    assert res.ok and res.attempts == 1
    req = llm.seen[0]
    assert req["temperature"] == 0.0 and req["max_tokens"] == settings.catalog_llm_max_tokens
    assert req["response_format"]["type"] == "json_schema" and req["stop"]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    syn = next(c for c in res.data["columns"] if c["name"] == "walk_idx")["synonyms"]
    assert "walkability index" in syn and "value" not in syn      # generic single words are dropped
    assert "walk" not in syn                                     # ... and short single words


def test_invented_columns_are_dropped_and_reported(llm):
    llm.script = [_draft(ALL + ["made_up_column"])]
    res = cl.draft_dataset(_ds())
    assert res.ok and all(c["name"] in ALL for c in res.data["columns"])
    assert any("invented" in n for n in res.notes)


def test_repair_pass_recovers_from_invalid_json(llm):
    llm.script = ["Sure! here you go", _draft(ALL)]
    res = cl.draft_dataset(_ds())
    assert res.ok and res.attempts == 2
    assert "rejected" in llm.seen[1]["messages"][-1]["content"]
    assert any("repair" in n for n in res.notes)


def test_partial_coverage_triggers_repair_then_fails_cleanly(llm):
    llm.script = [_draft(["geoid20"]), _draft(["geoid20"])]
    res = cl.draft_dataset(_ds())
    assert not res.ok and "1 of 5" in res.error


def test_server_without_json_schema_support_falls_back_to_prompt_only(llm):
    llm.script = [{"status": 400, "body": "response_format not supported"}, _draft(ALL)]
    res = cl.draft_dataset(_ds())
    assert res.ok
    assert "response_format" in llm.seen[0] and "response_format" not in llm.seen[1]


def test_falls_back_down_the_chain_when_the_draft_endpoint_is_down(llm, monkeypatch):
    main = _Server()
    try:
        main.script = [_draft(ALL)]
        monkeypatch.setattr(settings, "llama_server_base_url", main.url)
        monkeypatch.setattr(settings, "catalog_draft_base_url", "http://127.0.0.1:9/v1")   # nothing listens here
        cl.reset_router()
        res = cl.draft_dataset(_ds())
        assert res.ok and "chat-server" in res.endpoint
        st = cl.router().status()
        assert st["tiers"]["draft"][0]["ok"] is False and st["tiers"]["draft"][1]["ok"] is True
    finally:
        main.close()


def test_everything_offline_degrades_to_rules_and_never_raises(fresh_db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "catalog_draft_base_url", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(settings, "llama_server_base_url", "http://127.0.0.1:9/v1")
    cl.reset_router()
    from services import dataset_onboarding as ob
    (tmp_path / "d.csv").write_text("geoid20,natwalkind\n421010001001,5.5\n421010002002,7.5\n")
    ds = ob.create_dataset(tmp_path / "d.csv", "d.csv")
    out = ob.draft_with_model(ds["dataset_id"])
    assert out["draft"]["ok"] is False and out["draft"]["mode"] == "rules"
    assert next(s for s in out["pipeline"] if s["stage"] == "model_draft")["status"] == "skipped"
    assert out["columns"][1]["description"] == ""            # nothing was invented


def test_disabled_flag_short_circuits(llm, monkeypatch):
    monkeypatch.setattr(settings, "catalog_llm_enabled", False)
    res = cl.draft_dataset(_ds())
    assert not res.ok and "turned off" in res.error and not llm.seen


def test_selftest_scores_a_well_behaved_model_as_pass(llm):
    def auto(body):
        payload = json.loads(body["messages"][1]["content"])
        cols = [{"name": c["name"], "description": f"Holds {c['name']}.", "role": c["guessed_role"], "unit": "",
                 "synonyms": [f"{c['name'].replace('_', ' ')} measure"]} for c in payload["columns"]]
        return json.dumps({"title": "T", "description": "D.", "grain": "one row per thing", "domain": "other", "columns": cols})
    llm.auto = auto
    out = cl.selftest()
    assert out["verdict"] == "PASS", out
    assert out["summary"]["valid_rate"] == 1.0 and out["summary"]["role_agreement"] == 1.0


def test_selftest_fails_a_model_that_returns_garbage(llm):
    llm.auto = lambda body: "I cannot help with that."
    out = cl.selftest()
    assert out["verdict"] == "FAIL" and out["summary"]["valid_rate"] == 0.0


def test_judge_only_annotates_and_never_changes_the_numbers(llm):
    llm.script = [json.dumps({"items": [{"id": "0", "verdict": "plausible", "reason": "Both are 11-digit tract codes."},
                                        {"id": "9", "verdict": "plausible", "reason": "unknown id"}]})]
    props = [{"payload": {"left_table": "a", "left_expr": "tract_fips", "right_table": "houses", "right_expr": "tract_fips",
                          "cardinality": "many-to-one", "confidence": "high"},
              "evidence": {"stats": {"left_match_pct": 90.0, "right_covered_pct": 100.0}, "rule_text": "as is", "examples": []}}]
    res = cl.judge_links({"title": "t", "description": "d"}, props)
    assert res.ok
    assert props[0]["evidence"]["model_note"]["verdict"] == "plausible"
    assert props[0]["payload"]["confidence"] == "high" and props[0]["evidence"]["stats"]["left_match_pct"] == 90.0
