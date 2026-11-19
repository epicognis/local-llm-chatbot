"""HTTP-level tests (spec §13) against MockBackend and a temp SQLite store.
Drives the ASGI app in-process — no GPU, no network, no uvicorn."""

import asyncio
import json

import httpx
import pytest

from api.main import app
from config.settings import settings
from llm.backends.base import BackendError, LLMBackend
from llm.backends.mock_backend import MockBackend
from llm.registry import MODEL_REGISTRY
from orchestrator.orchestrator import Orchestrator
from session.store import SessionStore

MODEL = "Gemma 4 12B"


@pytest.fixture
async def store(tmp_path):
    s = SessionStore(f"sqlite:///{tmp_path / 'api.db'}")
    await s.init()
    return s


def install(store, backend):
    # Calling the ASGI app directly doesn't run lifespan, so wire state by hand.
    app.state.store = store
    app.state.backend = backend
    app.state.orchestrator = Orchestrator(backend, store)


@pytest.fixture
async def client(store):
    install(store, MockBackend(token_delay=0))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


def parse_sse(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            raw = line[6:]
            events.append(raw if raw == "[DONE]" else json.loads(raw))
    return events


async def test_health_ok_without_ollama(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_models_lists_registry(client):
    r = await client.get("/models")
    assert r.json() == MODEL_REGISTRY


async def test_ui_renders_with_default_model(client):
    r = await client.get("/ui")
    assert r.status_code == 200
    assert json.dumps(settings.DEFAULT_MODEL) in r.text


async def test_chat_streams_and_persists_exchange(client):
    r = await client.post("/chat", json={"session_id": "s1", "message": "hello", "model": MODEL})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    tokens = [e for e in events[:-1] if e["type"] == "token"]
    assert len(tokens) > 3
    assert "".join(e["text"] for e in tokens).startswith('[MOCK] You said: "hello"')

    turns = (await client.get("/sessions/s1/turns")).json()
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert (await client.get("/sessions")).json()[0]["session_id"] == "s1"


async def test_unknown_model_is_error_event_and_creates_nothing(client):
    r = await client.post("/chat", json={"session_id": "s2", "message": "hi", "model": "Nope 9B"})
    events = parse_sse(r.text)
    assert events[0]["type"] == "error"
    assert "Unknown model 'Nope 9B'" in events[0]["text"]
    assert events[-1] == "[DONE]"
    assert (await client.get("/sessions")).json() == []


async def test_empty_message_rejected(client):
    r = await client.post("/chat", json={"session_id": "s", "message": ""})
    assert r.status_code == 422


class FailingBackend(MockBackend):
    def _chat_stream(self, messages, model, **opts):
        async def gen():
            raise BackendError("Model 'x' is not available in Ollama. Pull it first: ollama pull x")
            yield  # pragma: no cover
        return gen()


async def test_backend_error_is_not_persisted_as_assistant_text(store):
    install(store, FailingBackend())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/chat", json={"session_id": "s3", "message": "hi", "model": MODEL})
    events = parse_sse(r.text)
    assert events[0] == {"type": "error", "text": "Model 'x' is not available in Ollama. Pull it first: ollama pull x"}
    assert await store.get_turns("s3") == []


class SlowBackend(LLMBackend):
    """Streams forever until cancelled; records whether the upstream was torn down."""

    def __init__(self):
        self.cancelled = asyncio.Event()

    def _chat_stream(self, messages, model, **opts):
        return self._gen()

    async def _gen(self):
        try:
            i = 0
            while True:
                yield f"tok{i} "
                i += 1
                await asyncio.sleep(0.01)
        finally:
            self.cancelled.set()

    async def _chat_complete(self, messages, model, **opts):
        return ""


async def test_client_disconnect_aborts_upstream_and_keeps_partial_reply(store):
    backend = SlowBackend()
    install(store, backend)

    body = json.dumps({"session_id": "s4", "message": "go", "model": MODEL}).encode()
    disconnect = asyncio.Event()
    received_chunks = 0

    async def receive():
        nonlocal body
        if body is not None:
            msg, body = {"type": "http.request", "body": body, "more_body": False}, None
            return msg
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal received_chunks
        if message["type"] == "http.response.body" and b"tok" in message.get("body", b""):
            received_chunks += 1
            if received_chunks >= 5:
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

    # The partial reply is persisted (shielded from the cancellation) and paired
    # with its user turn.
    for _ in range(50):
        turns = await store.get_turns("s4")
        if turns:
            break
        await asyncio.sleep(0.02)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].content.startswith("tok0 tok1 ")
