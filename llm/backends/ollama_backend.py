"""Ollama backend — streaming over the /api/chat endpoint.

Why /api/chat and not /api/generate: Ollama applies each model's own chat template
to role-tagged messages, so we never hand-maintain per-model prompt formatting.

Hard-won rules baked in here (see build spec §6):

1. ``num_ctx`` is sent explicitly per request. Ollama defaults it to 4096
   regardless of the model's real window, silently capping long context.

2. Reasoning models split output across ``message.content`` (the answer) and
   ``message.thinking`` (chain-of-thought), inconsistently:
     - Qwen3 can route everything to ``thinking`` (content empty).
     - GPT-OSS fills BOTH per chunk, and can't fully disable reasoning
       (the registry sends ``think: "low"`` for it).
   So we only ever stream ``content`` as the answer. ``thinking`` is surfaced as
   a liveness heartbeat and buffered — emitted as text only as a last-resort
   fallback if the turn finished with no content at all. If the turn ran out of
   ``num_predict`` mid-reasoning, the buffer is half-finished chain-of-thought,
   so that's reported as an error instead of being shown.

3. Errors are surfaced as :class:`BackendError` with an actionable message
   (daemon not running, model not pulled -> ``ollama pull <tag>``, mid-stream
   errors) — never streamed as if the model had said them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import structlog

from .base import BackendError, ChatEvent, LLMBackend

log = structlog.get_logger(__name__)


def _seconds(ns: int | None) -> float | None:
    return ns / 1e9 if ns else None


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # One pooled client for the app's lifetime. No read timeout: generation
        # (or a cold load of a large num_ctx) can legitimately take minutes.
        # Cancellation is handled by the caller closing the chat() generator,
        # which exits the stream context and aborts the upstream request.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _options(
        num_ctx: int | None,
        num_gpu: int | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        top_k: int | None,
        extra: dict,
    ) -> dict:
        opts: dict = {}
        if num_ctx is not None:
            opts["num_ctx"] = num_ctx  # CRITICAL: else Ollama caps at 4096
        if num_gpu is not None:
            opts["num_gpu"] = num_gpu  # layers offloaded to GPU; 0 = full CPU
        if temperature is not None:
            opts["temperature"] = temperature
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if top_p is not None:
            opts["top_p"] = top_p
        if top_k is not None:
            opts["top_k"] = top_k
        # Pass-through for any other Ollama option (seed, repeat_penalty, ...).
        opts.update(extra.get("options", {}))
        return opts

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

    async def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        think: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        keep_alive: int | str | None = None,
        **opts,
    ) -> AsyncIterator[ChatEvent]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": self._options(num_ctx, num_gpu, temperature, max_tokens, top_p, top_k, opts),
        }
        if think is not None:
            payload["think"] = think
        # keep_alive: how long Ollama holds the model resident after this request
        # (-1 = indefinitely, 0 = unload now, else seconds or a duration string).
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        content_seen = False
        thinking_buf: list[str] = []
        final: dict = {}

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                await self._raise_for_status(resp, model)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("ollama_bad_json_line", line=line[:200])
                        continue
                    if "error" in chunk:
                        raise BackendError(f"Ollama error: {chunk['error']}")

                    msg = chunk.get("message") or {}
                    content = msg.get("content") or ""
                    thinking = msg.get("thinking") or ""
                    if content:
                        content_seen = True
                        yield ChatEvent(delta=content)
                    elif thinking:
                        thinking_buf.append(thinking)
                        yield ChatEvent(thinking_delta=thinking)

                    if chunk.get("done"):
                        final = chunk
                        break
        except httpx.ConnectError:
            raise self._connect_error() from None
        except httpx.HTTPError as exc:
            raise BackendError(f"Lost connection to Ollama mid-stream: {exc!r}") from None

        done_reason = final.get("done_reason")
        if not content_seen:
            if thinking_buf and done_reason == "length":
                raise BackendError(
                    "The model used its whole token budget reasoning and never got to an "
                    "answer. Try again with a larger Max tokens, or turn Thinking mode off."
                )
            if thinking_buf:
                # A turn that produced ONLY (finished) reasoning still yields a
                # visible answer rather than dead air.
                log.info("ollama_thinking_fallback", model=model)
                yield ChatEvent(delta="".join(thinking_buf), from_thinking=True)
            else:
                # Ollama can return a clean "done" with zero output, e.g. when a
                # large-num_ctx cold load doesn't finish in time.
                raise BackendError(
                    "The model returned an empty response — it may still be loading "
                    "(large context windows can take a while to warm up). Please try again."
                )

        yield ChatEvent(
            done=True,
            prompt_eval_count=final.get("prompt_eval_count"),
            eval_count=final.get("eval_count"),
            eval_duration_s=_seconds(final.get("eval_duration")),
            load_duration_s=_seconds(final.get("load_duration")),
            done_reason=done_reason,
        )

    # ── residency / loading ───────────────────────────────────────────

    async def _ps(self) -> list[dict]:
        resp = await self._client.get("/api/ps", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models") or []

    async def version(self) -> str:
        """Daemon version; raises if Ollama isn't reachable (used by /health)."""
        resp = await self._client.get("/api/version", timeout=3)
        resp.raise_for_status()
        return resp.json().get("version", "?")

    async def resident_models(self) -> list[str]:
        try:
            return [t for m in await self._ps() if (t := m.get("name") or m.get("model"))]
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("ollama_ps_failed", error=str(exc))
            return []

    async def warm_state(self, model: str, num_ctx: int | None = None) -> str:
        # Best effort: any hiccup reports "warm" so chat is never blocked by it.
        try:
            for m in await self._ps():
                if m.get("name") == model:
                    loaded_ctx = m.get("context_length")
                    if num_ctx and loaded_ctx and loaded_ctx != num_ctx:
                        return "reload"
                    return "warm"
            tags = await self._client.get("/api/tags", timeout=5)
            if tags.status_code == 200 and not any(
                m.get("name") == model for m in tags.json().get("models", [])
            ):
                return "missing"
            return "cold"
        except Exception:  # noqa: BLE001
            return "warm"

    async def _unload(self, tag: str, timeout: float | None = 30) -> None:
        await self._client.post(
            "/api/generate", json={"model": tag, "keep_alive": 0}, timeout=timeout
        )

    async def unload_others(self, keep_model: str) -> list[str]:
        """Single-model enforcement: unload every resident model except ``keep_model``."""
        unloaded: list[str] = []
        try:
            for tag in await self.resident_models():
                if tag != keep_model:
                    await self._unload(tag)
                    unloaded.append(tag)
        except httpx.HTTPError as exc:
            log.warning("ollama_purge_failed", error=str(exc))
        if unloaded:
            log.info("ollama_unloaded_others", kept=keep_model, unloaded=unloaded)
        return unloaded

    async def unload_all(self) -> dict:
        """Unload every resident model, then re-check ``/api/ps`` to confirm.

        Bounded: a model that is mid-generation can't unload until its in-flight
        request finishes, so each unload has a short timeout — better to report
        "still_resident" quickly than block the caller for minutes.
        """
        before = await self.resident_models()
        for tag in before:
            try:
                await self._unload(tag, timeout=12)
            except httpx.HTTPError as exc:
                log.warning("ollama_unload_failed", model=tag, error=str(exc))
        if before:
            # Let the runner subprocess(es) exit and release VRAM before re-reading.
            await asyncio.sleep(0.5)
        after = await self.resident_models()
        unloaded = [t for t in before if t not in after]
        log.info("ollama_unloaded_all", before=before, unloaded=unloaded, still_resident=after)
        return {"before": before, "unloaded": unloaded, "still_resident": after}

    async def load(
        self,
        model: str,
        *,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        keep_alive: int | str | None = None,
        purge_others: bool = False,
    ) -> dict:
        """Load a model ahead of the first turn via an empty ``/api/generate``.

        ``num_ctx`` / ``num_gpu`` are passed so the runner is sized and placed to
        match generation (no reload on the first turn).
        """
        unloaded = await self.unload_others(model) if purge_others else []
        payload: dict = {"model": model}
        options: dict = {}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            resp = await self._client.post("/api/generate", json=payload)
        except httpx.ConnectError:
            raise self._connect_error() from None
        await self._raise_for_status(resp, model)
        data = resp.json() if resp.content else {}
        log.info("ollama_model_loaded", model=model, num_ctx=num_ctx, num_gpu=num_gpu, unloaded=unloaded)
        return {"status": "loaded", "model": model, "unloaded": unloaded,
                "done_reason": data.get("done_reason")}
