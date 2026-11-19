import json

import httpx
import structlog

from llm.backends.base import BackendError, LLMBackend, StreamStatus

log = structlog.get_logger()

_DURATION_FIELDS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")


def _fill_stats(stats: dict | None, chunk: dict) -> None:
    """Copy Ollama's end-of-turn counters into the caller's stats sink.
    Durations arrive in nanoseconds; they're stored as seconds with an _s suffix."""
    if stats is None:
        return
    for key in ("prompt_eval_count", "eval_count", "done_reason"):
        if key in chunk:
            stats[key] = chunk[key]
    for key in _DURATION_FIELDS:
        if chunk.get(key) is not None:
            stats[f"{key}_s"] = chunk[key] / 1e9


class OllamaBackend(LLMBackend):
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        # One pooled client for the app's lifetime. No read timeout: a cold load
        # of a large num_ctx can legitimately take minutes before the first byte.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _payload(messages: list[dict], model: str, stream: bool, opts: dict) -> tuple[dict, dict | None]:
        opts = dict(opts)
        stats = opts.pop("stats", None)
        think = opts.pop("think", None)
        keep_alive = opts.pop("keep_alive", None)
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if opts:
            payload["options"] = opts
        return payload, stats

    @staticmethod
    async def _raise_for_status(response: httpx.Response, model: str) -> None:
        if response.is_success:
            return
        await response.aread()
        try:
            detail = response.json().get("error") or response.text
        except ValueError:
            detail = response.text
        if response.status_code == 404 and "not found" in detail.lower():
            raise BackendError(
                f"Model '{model}' is not available in Ollama. Pull it first: ollama pull {model}"
            )
        raise BackendError(f"Ollama returned HTTP {response.status_code}: {detail}")

    def _connect_error(self) -> BackendError:
        return BackendError(f"Cannot connect to Ollama at {self.base_url}. Is it running?")

    # ── chat ───────────────────────────────────────────────────────────

    def _chat_stream(self, messages: list[dict], model: str, **opts):
        return self._stream_from_ollama(messages, model, **opts)

    async def _stream_from_ollama(self, messages: list[dict], model: str, **opts):
        payload, stats = self._payload(messages, model, True, opts)

        content_emitted = False
        thinking_buffer: list[str] = []
        done_reason = None

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                await self._raise_for_status(response, model)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise BackendError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    # Reasoning models (spec §6): GPT-OSS fills both `content` and
                    # `thinking` per chunk; Qwen3 may fill only `thinking`. Only
                    # `content` is ever streamed live — `thinking` is buffered as
                    # a last-resort fallback.
                    content = msg.get("content", "")
                    if content:
                        content_emitted = True
                        yield content
                    elif msg.get("thinking"):
                        if not thinking_buffer:
                            yield StreamStatus("Thinking…")
                        thinking_buffer.append(msg["thinking"])
                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason")
                        _fill_stats(stats, chunk)
                        log.debug(
                            "ollama_turn_done",
                            model=model,
                            prompt_eval_count=chunk.get("prompt_eval_count"),
                            eval_count=chunk.get("eval_count"),
                            done_reason=chunk.get("done_reason"),
                        )
                        break
        except httpx.ConnectError:
            raise self._connect_error() from None
        except httpx.HTTPError as exc:
            raise BackendError(f"Lost connection to Ollama mid-stream: {exc!r}") from None

        if not content_emitted:
            if thinking_buffer and done_reason == "length":
                # Reasoning used up num_predict before any answer: the buffer is
                # half-finished chain-of-thought, not a reply — don't show it.
                raise BackendError(
                    "The model used its whole token budget reasoning and never got "
                    "to an answer. Try again with a larger max_tokens."
                )
            if thinking_buffer:
                yield "".join(thinking_buffer)
            else:
                # Ollama can return a clean "done" with zero output, e.g. when a
                # large-num_ctx cold load doesn't finish in time.
                raise BackendError(
                    "The model returned an empty response — it may still be loading "
                    "(large context windows can take a while to warm up). Please try again."
                )

    async def _chat_complete(self, messages: list[dict], model: str, **opts) -> str:
        payload, stats = self._payload(messages, model, False, opts)
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectError:
            raise self._connect_error() from None
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc!r}") from None
        await self._raise_for_status(response, model)
        data = response.json()
        if "error" in data:
            raise BackendError(f"Ollama error: {data['error']}")
        _fill_stats(stats, data)
        msg = data.get("message") or {}
        text = msg.get("content") or msg.get("thinking") or ""
        if not text:
            raise BackendError("The model returned an empty response.")
        return text

    async def warm_state(self, model: str, num_ctx: int | None = None) -> str:
        # Best effort: any hiccup reports "warm" so chat is never blocked by it.
        try:
            resp = await self._client.get("/api/ps", timeout=3)
            if resp.status_code != 200:
                return "warm"
            for m in resp.json().get("models", []):
                if m.get("name") == model:
                    loaded_ctx = m.get("context_length")
                    if num_ctx and loaded_ctx and loaded_ctx != num_ctx:
                        return "reload"
                    return "warm"
            tags = await self._client.get("/api/tags", timeout=3)
            if tags.status_code == 200 and not any(
                m.get("name") == model for m in tags.json().get("models", [])
            ):
                return "missing"
            return "cold"
        except Exception:
            return "warm"

    async def loaded_models(self) -> list[str]:
        resp = await self._client.get("/api/ps", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
