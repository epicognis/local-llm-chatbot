"""OllamaBackend against a fake Ollama (httpx.MockTransport) — no GPU, no network.
Covers spec §6 / §13: thinking-field handling, num_ctx passthrough, error surfacing."""

import json

import httpx
import pytest

from llm.backends.base import BackendError, StreamStatus
from llm.backends.ollama_backend import OllamaBackend


def jsonl(*chunks: dict) -> bytes:
    return "".join(json.dumps(c) + "\n" for c in chunks).encode()


def make_backend(handler) -> tuple[OllamaBackend, list[dict]]:
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.content:
            seen.append(json.loads(request.content))
        return handler(request)

    return OllamaBackend("http://ollama.test", transport=httpx.MockTransport(wrapped)), seen


async def collect(backend, statuses: list | None = None, **opts) -> str:
    parts = []
    async for tok in backend.chat([{"role": "user", "content": "hi"}], "m", stream=True, **opts):
        if isinstance(tok, StreamStatus):
            if statuses is not None:
                statuses.append(str(tok))
            continue
        parts.append(tok)
    return "".join(parts)


DONE = {"done": True, "done_reason": "stop", "prompt_eval_count": 42, "eval_count": 3,
        "eval_duration": 1_500_000_000, "load_duration": 250_000_000}


async def test_streams_only_content_never_thinking():
    # GPT-OSS style: both fields populated in the same chunks.
    body = jsonl(
        {"message": {"content": "Hel", "thinking": "secret plan"}},
        {"message": {"content": "lo", "thinking": " more secrets"}},
        DONE,
    )
    backend, _ = make_backend(lambda r: httpx.Response(200, content=body))
    out = await collect(backend)
    assert out == "Hello"
    assert "secret" not in out


async def test_thinking_only_model_falls_back_to_buffered_thinking():
    # Qwen3 style: everything in `thinking`, `content` empty all turn.
    body = jsonl(
        {"message": {"content": "", "thinking": "The answer "}},
        {"message": {"content": "", "thinking": "is 4."}},
        DONE,
    )
    backend, _ = make_backend(lambda r: httpx.Response(200, content=body))
    assert await collect(backend) == "The answer is 4."


async def test_empty_turn_raises_instead_of_yielding_text():
    backend, _ = make_backend(lambda r: httpx.Response(200, content=jsonl(DONE)))
    with pytest.raises(BackendError, match="empty response"):
        await collect(backend)


async def test_payload_carries_num_ctx_think_keep_alive_and_fills_stats():
    body = jsonl({"message": {"content": "ok"}}, DONE)
    backend, seen = make_backend(lambda r: httpx.Response(200, content=body))
    stats: dict = {}
    await collect(backend, num_ctx=65536, think=False, keep_alive="30m",
                  temperature=0.2, stats=stats)
    payload = seen[0]
    assert payload["options"] == {"num_ctx": 65536, "temperature": 0.2}
    assert payload["think"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["stream"] is True
    assert "stats" not in payload and "stats" not in payload["options"]
    assert stats["prompt_eval_count"] == 42
    assert stats["eval_count"] == 3
    assert stats["eval_duration_s"] == pytest.approx(1.5)


async def test_think_omitted_when_not_requested():
    body = jsonl({"message": {"content": "ok"}}, DONE)
    backend, seen = make_backend(lambda r: httpx.Response(200, content=body))
    await collect(backend, num_ctx=8192)
    assert "think" not in seen[0]


async def test_model_not_pulled_gives_actionable_error():
    backend, _ = make_backend(
        lambda r: httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"})
    )
    with pytest.raises(BackendError, match="ollama pull m"):
        await collect(backend)


async def test_mid_stream_error_line_raises():
    body = jsonl({"message": {"content": "par"}}, {"error": "CUDA out of memory"})
    backend, _ = make_backend(lambda r: httpx.Response(200, content=body))
    with pytest.raises(BackendError, match="CUDA out of memory"):
        await collect(backend)


async def test_connection_refused_is_a_backend_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    backend, _ = make_backend(refuse)
    with pytest.raises(BackendError, match="Cannot connect"):
        await collect(backend)


async def test_summarize_passes_runtime_options():
    backend, seen = make_backend(
        lambda r: httpx.Response(200, json={"message": {"content": "summary"}, "done": True})
    )
    out = await backend.summarize("long transcript", "m", num_ctx=40960, think=False)
    assert out == "summary"
    assert seen[0]["stream"] is False
    assert seen[0]["options"]["num_ctx"] == 40960
    assert seen[0]["think"] is False


async def test_thinking_phase_reports_status_once():
    body = jsonl(
        {"message": {"content": "", "thinking": "a"}},
        {"message": {"content": "", "thinking": "b"}},
        {"message": {"content": "Answer"}},
        DONE,
    )
    backend, _ = make_backend(lambda r: httpx.Response(200, content=body))
    statuses: list = []
    assert await collect(backend, statuses=statuses) == "Answer"
    assert statuses == ["Thinking…"]


async def test_reasoning_that_exhausts_budget_is_not_shown_as_answer():
    body = jsonl(
        {"message": {"content": "", "thinking": "half-finished chain of thought"}},
        {**DONE, "done_reason": "length"},
    )
    backend, _ = make_backend(lambda r: httpx.Response(200, content=body))
    with pytest.raises(BackendError, match="whole token budget reasoning"):
        await collect(backend)


def ps_tags_handler(ps_models, tag_names):
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": ps_models})
        return httpx.Response(200, json={"models": [{"name": n} for n in tag_names]})
    return handler


async def test_warm_state():
    resident = [{"name": "m", "context_length": 4096}]
    backend, _ = make_backend(ps_tags_handler(resident, ["m", "other"]))
    assert await backend.warm_state("m", 4096) == "warm"
    assert await backend.warm_state("m", 65536) == "reload"   # num_ctx change forces reload
    assert await backend.warm_state("other", 8192) == "cold"
    assert await backend.warm_state("absent", 8192) == "missing"
