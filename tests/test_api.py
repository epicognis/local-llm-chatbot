"""End-to-end API tests against the MockBackend (no GPU, no network)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import Settings
from llm.backends.base import BackendError, ChatEvent, LLMBackend
from llm.backends.mock_backend import MockBackend


def make_settings(tmp_path, **kw):
    return Settings(llm_backend="mock", session_db_url=f"sqlite:///{tmp_path / 'sessions.db'}",
                    context_level=0, **kw)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


def events(text: str) -> list:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            raw = line[6:]
            out.append(raw if raw == "[DONE]" else json.loads(raw))
    return out


def chat(client, **body):
    body.setdefault("session_id", "s")
    body.setdefault("message", "ping")
    with client.stream("POST", "/chat", json=body) as res:
        assert res.status_code == 200
        return events("".join(res.iter_text()))


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200 and res.json()["status"] == "ok"


def test_models_listed_in_dropdown_order(client):
    models = client.get("/models").json()
    names = [m["name"] for m in models]
    assert names[0] == "Gemma 4 12B"  # default first
    assert names.index("Llama 3.1 8B") < names.index("Qwen3 14B")
    assert all({"name", "model", "default_ctx", "supports_think"} <= m.keys() for m in models)


def test_load_resolves_tag_and_handles_unknown(client):
    ok = client.post("/load", json={"model": "Gemma 4 12B"}).json()
    assert ok["tag"] == "gemma4:12b" and ok["status"] == "noop"
    bad = client.post("/load", json={"model": "Nope"})
    assert bad.status_code == 200 and bad.json()["status"] == "error"


def test_chat_streams_status_deltas_stats_done(client):
    evs = chat(client, model="GPT-OSS 20B")
    assert evs[0] == {"status": "prefill"}
    text = "".join(e["delta"] for e in evs if isinstance(e, dict) and "delta" in e)
    assert text.endswith("You said: ping")
    stats = next(e["stats"] for e in evs if isinstance(e, dict) and "stats" in e)
    assert stats["eval_count"]
    assert evs[-1] == "[DONE]"


def test_thinking_heartbeat_never_leaks_reasoning_text(client):
    evs = chat(client, model="GPT-OSS 20B", options={"think": True})
    assert any(isinstance(e, dict) and "thinking" in e for e in evs)
    assert "think about" not in json.dumps(evs)


def test_chat_persists_and_session_reloads(client):
    chat(client, session_id="p", message="remember this", model="GPT-OSS 20B")
    data = client.get("/sessions/p").json()
    assert data["model"] == "GPT-OSS 20B"
    assert data["turns"][0] == {"role": "user", "content": "remember this"}
    assert data["turns"][1]["role"] == "assistant"


def test_unknown_model_is_error_event_and_creates_nothing(client):
    evs = chat(client, session_id="u", model="Nope 9B")
    assert "Unknown model 'Nope 9B'" in evs[0]["error"]
    assert evs[-1] == "[DONE]"
    assert client.get("/sessions").json() == []


def test_invalid_requests_rejected(client):
    assert client.post("/chat", json={"session_id": "s", "message": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "message": "x",
                                      "options": {"num_gpu": -1}}).status_code == 422


def test_force_cpu_num_gpu_reaches_backend(client):
    chat(client, model="GPT-OSS 20B", options={"num_gpu": 0, "num_ctx": 4096})
    call = client.app.state.backend.calls[-1]
    assert call["num_gpu"] == 0 and call["num_ctx"] == 4096


def test_rename_and_delete_session(client):
    chat(client, session_id="r")
    assert client.patch("/sessions/r", json={"title": "My chat"}).json()["title"] == "My chat"
    assert any(s["id"] == "r" and s["title"] == "My chat" for s in client.get("/sessions").json())
    assert client.delete("/sessions/r").json()["deleted"] is True
    assert client.get("/sessions/r").status_code == 404


def test_rename_delete_unknown_session_404(client):
    assert client.patch("/sessions/ghost", json={"title": "x"}).status_code == 404
    assert client.delete("/sessions/ghost").status_code == 404


def test_unload_others_enforced_once_per_model(client):
    backend = client.app.state.backend
    calls = []
    orig = backend.unload_others

    async def counting(keep_model):
        calls.append(keep_model)
        return await orig(keep_model)

    backend.unload_others = counting
    chat(client, model="GPT-OSS 20B")
    chat(client, model="GPT-OSS 20B")          # same model -> skipped
    assert calls == ["gpt-oss:20b"]
    chat(client, model="Llama 3.1 8B")          # switch -> enforce again
    assert calls == ["gpt-oss:20b", "llama3.1:8b"]


def test_unload_all_purges(client):
    body = client.post("/unload").json()
    assert body == {"status": "ok", "before": [], "unloaded": [], "still_resident": []}


def test_monitor_endpoint_always_200(client):
    res = client.get("/monitor")
    assert res.status_code == 200 and "available" in res.json()


def test_ui_renders_with_controls_and_no_stale_cache(client):
    res = client.get("/ui")
    assert res.status_code == 200 and res.headers["cache-control"] == "no-store"
    ui = res.text
    for marker in ('id="cfg-auto-load"', 'id="purge-gpu"', 'id="sidebar-toggle"',
                   'id="busy-canvas"', "/static/lissajous.js",
                   'fetch("/sessions", { cache: "no-store" })'):
        assert marker in ui
    assert json.dumps("Gemma 4 12B") in ui
    assert client.get("/static/lissajous.js").status_code == 200


def test_root_redirects_to_ui(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (307, 308) and res.headers["location"] == "/ui"


# ── failure + cancellation semantics ──────────────────────────────────────


class FailingBackend(MockBackend):
    async def chat(self, messages, model, **kw):
        raise BackendError("Model 'x' is not available in Ollama. Pull it first: ollama pull x")
        yield  # pragma: no cover


def test_backend_error_is_event_and_nothing_is_saved(tmp_path):
    with TestClient(create_app(make_settings(tmp_path), backend=FailingBackend())) as c:
        evs = chat(c, session_id="f", model="Gemma 4 12B")
        assert {"error": "Model 'x' is not available in Ollama. Pull it first: ollama pull x"} in evs
        assert c.app.state.store.get_turns("f") == []
        assert c.get("/sessions").json() == []


class SlowBackend(LLMBackend):
    """Streams forever until cancelled; records whether the upstream was torn down."""

    def __init__(self):
        self.cancelled = asyncio.Event()

    async def chat(self, messages, model, **kw):
        try:
            i = 0
            while True:
                yield ChatEvent(delta=f"tok{i} ")
                i += 1
                await asyncio.sleep(0.01)
        finally:
            self.cancelled.set()


async def test_client_disconnect_aborts_upstream_and_keeps_partial_reply(tmp_path):
    backend = SlowBackend()
    app = create_app(make_settings(tmp_path), backend=backend)

    body = json.dumps({"session_id": "d", "message": "go", "model": "Gemma 4 12B"}).encode()
    disconnect = asyncio.Event()
    sent = 0

    async def receive():
        nonlocal body
        if body is not None:
            msg, body = {"type": "http.request", "body": body, "more_body": False}, None
            return msg
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal sent
        if message["type"] == "http.response.body" and b"tok" in message.get("body", b""):
            sent += 1
            if sent >= 5:
                disconnect.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/chat",
        "raw_path": b"/chat", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"t")],
        "client": ("127.0.0.1", 1), "server": ("t", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    await asyncio.wait_for(backend.cancelled.wait(), timeout=5)

    store = app.state.store
    for _ in range(50):
        turns = store.get_turns("d")
        if turns:
            break
        await asyncio.sleep(0.02)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["content"].startswith("tok0 tok1 ")
