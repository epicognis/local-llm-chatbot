"""Backend behavior — MockBackend directly, OllamaBackend against a fake Ollama
(httpx.MockTransport). Pins the spec §6 rules without a GPU: thinking-field
handling, num_ctx passthrough, actionable errors, residency checks."""

import json

import httpx
import pytest

from llm.backends.base import BackendError
from llm.backends.mock_backend import MockBackend
from llm.backends.ollama_backend import OllamaBackend


def jsonl(*chunks: dict) -> str:
    return "\n".join(json.dumps(c) for c in chunks)


DONE = {"done": True, "done_reason": "stop", "prompt_eval_count": 42, "eval_count": 3,
        "eval_duration": 1_500_000_000, "load_duration": 250_000_000}


def make_backend(handler):
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.content:
            seen.append({"path": request.url.path, **json.loads(request.content)})
        return handler(request)

    return OllamaBackend("http://ollama.test", transport=httpx.MockTransport(wrapped)), seen


def chat_handler(body: str, status: int = 200):
    return lambda request: httpx.Response(status, text=body)


async def run(backend, **opts):
    content, thinking, final = [], [], None
    async for ev in backend.chat([{"role": "user", "content": "hi"}], "m", **opts):
        if ev.delta:
            content.append(ev.delta)
        if ev.thinking_delta:
            thinking.append(ev.thinking_delta)
        if ev.done:
            final = ev
    return "".join(content), thinking, final


async def test_mock_streams_and_finishes():
    backend = MockBackend()
    out, _, final = await run(backend)
    assert out.endswith("You said: hi")
    assert final.prompt_eval_count is not None and final.eval_count


async def test_streams_only_content_when_both_fields_present():
    # GPT-OSS style: every chunk has BOTH thinking and content.
    body = jsonl({"message": {"content": "Hel", "thinking": "secret plan"}},
                 {"message": {"content": "lo", "thinking": " more"}}, DONE)
    backend, _ = make_backend(chat_handler(body))
    out, thinking, final = await run(backend)
    assert out == "Hello"
    assert "secret" not in out
    assert final.prompt_eval_count == 42
    assert final.eval_duration_s == pytest.approx(1.5)
    assert final.load_duration_s == pytest.approx(0.25)


async def test_thinking_only_turn_is_heartbeat_then_fallback():
    # Qwen3 style: everything in `thinking`, `content` empty all turn.
    body = jsonl({"message": {"content": "", "thinking": "The answer "}},
                 {"message": {"content": "", "thinking": "is 4."}}, DONE)
    backend, _ = make_backend(chat_handler(body))
    out, thinking, _ = await run(backend)
    assert thinking == ["The answer ", "is 4."]  # heartbeat (counted, never shown)
    assert out == "The answer is 4."             # last-resort fallback


async def test_reasoning_that_exhausts_budget_is_an_error_not_an_answer():
    body = jsonl({"message": {"content": "", "thinking": "half-finished chain of thought"}},
                 {**DONE, "done_reason": "length"})
    backend, _ = make_backend(chat_handler(body))
    with pytest.raises(BackendError, match="whole token budget reasoning"):
        await run(backend)


async def test_empty_turn_raises_instead_of_yielding_text():
    backend, _ = make_backend(chat_handler(jsonl(DONE)))
    with pytest.raises(BackendError, match="empty response"):
        await run(backend)


async def test_payload_carries_num_ctx_think_and_decoding_options():
    backend, seen = make_backend(chat_handler(jsonl({"message": {"content": "ok"}}, DONE)))
    await run(backend, num_ctx=40960, num_gpu=0, think="low", temperature=0.3,
              max_tokens=256, top_p=0.85, top_k=20, keep_alive=-1)
    p = seen[0]
    assert p["options"] == {"num_ctx": 40960, "num_gpu": 0, "temperature": 0.3,
                            "num_predict": 256, "top_p": 0.85, "top_k": 20}
    assert p["think"] == "low"
    assert p["keep_alive"] == -1  # top-level field, not an option
    assert p["stream"] is True


async def test_think_omitted_for_models_without_reasoning():
    backend, seen = make_backend(chat_handler(jsonl({"message": {"content": "ok"}}, DONE)))
    await run(backend, num_ctx=8192, think=None)
    assert "think" not in seen[0]


async def test_model_not_pulled_gives_actionable_error():
    backend, _ = make_backend(
        lambda r: httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"}))
    with pytest.raises(BackendError, match="ollama pull m"):
        await run(backend)


async def test_mid_stream_error_line_raises():
    body = jsonl({"message": {"content": "par"}}, {"error": "CUDA out of memory"})
    backend, _ = make_backend(chat_handler(body))
    with pytest.raises(BackendError, match="CUDA out of memory"):
        await run(backend)


async def test_connection_refused_is_a_backend_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    backend, _ = make_backend(refuse)
    with pytest.raises(BackendError, match="Cannot connect"):
        await run(backend)


def residency(ps_models, tag_names):
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": ps_models})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in tag_names]})
        return httpx.Response(200, json={"done": True, "done_reason": "load"})
    return handler


async def test_warm_state():
    backend, _ = make_backend(residency([{"name": "m", "context_length": 4096}], ["m", "other"]))
    assert await backend.warm_state("m", 4096) == "warm"
    assert await backend.warm_state("m", 65536) == "reload"   # num_ctx change forces a reload
    assert await backend.warm_state("other", 8192) == "cold"
    assert await backend.warm_state("absent", 8192) == "missing"


async def test_load_purges_others_and_matches_num_ctx():
    backend, seen = make_backend(residency([{"name": "a"}, {"name": "keep"}], ["a", "keep"]))
    result = await backend.load("keep", num_ctx=32768, num_gpu=0, keep_alive=-1, purge_others=True)
    assert result["status"] == "loaded" and result["unloaded"] == ["a"]
    unload, load = seen
    assert unload == {"path": "/api/generate", "model": "a", "keep_alive": 0}
    assert load == {"path": "/api/generate", "model": "keep",
                    "options": {"num_ctx": 32768, "num_gpu": 0}, "keep_alive": -1}
