"""POST /chat — streamed completion over Server-Sent Events.

SSE wire format: each orchestrator event dict is sent as one ``data:`` line,
then a terminal ``data: [DONE]``. Event kinds:
  ``{"status": "loading"|"summarizing", "text": ...}``  slow work before the reply
  ``{"status": "prefill"}``  request in flight, no token yet (UI shows "waiting")
  ``{"delta": "..."}``       a content chunk to append to the answer
  ``{"thinking": <int>}``    running reasoning-token count (liveness heartbeat)
  ``{"stats": {...}}``       terminal per-turn metrics (ttft_s, eval_count, tok_s, …)
  ``{"error": "..."}``       the turn failed; nothing was saved to history
While the model is silent, an SSE comment (``: keepalive``) is sent every few
seconds.

Cancellation: the turn runs in its own task. The response generator polls for
a client disconnect and cancels that task, which closes the backend stream and
aborts the upstream Ollama call; the orchestrator still persists any partial
reply.
"""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm.backends.base import BackendError
from llm.registry import UnknownModelError
from orchestrator.context_budget import ContextOverflowError

log = structlog.get_logger(__name__)
router = APIRouter()

KEEPALIVE_S = 5.0
DISCONNECT_POLL_S = 1.0
_DONE = object()
_EXPECTED_ERRORS = (BackendError, UnknownModelError, ContextOverflowError)


class ChatOptions(BaseModel):
    # External decoding knobs from the UI's LLM Config panel. All optional: when
    # None the orchestrator uses the configured defaults (num_ctx: the model's
    # registry default_ctx).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    num_ctx: int | None = Field(default=None, ge=256, le=1048576)
    # Layers to offload to the GPU. 0 forces full CPU execution ("Force CPU").
    num_gpu: int | None = Field(default=None, ge=0, le=999)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=1000)
    # Ollama keep_alive: -1 = resident indefinitely, 0 = unload now, a positive
    # number of seconds, or a duration string ("10m").
    keep_alive: int | str | None = None
    think: bool | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    model: str | None = None
    options: ChatOptions = Field(default_factory=ChatOptions)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    orchestrator = request.app.state.orchestrator
    model_name = req.model or request.app.state.settings.default_model
    options = req.options.model_dump(exclude_none=True)
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for event in orchestrator.generate(
                session_id=req.session_id,
                message=req.message,
                model_name=model_name,
                options=options,
            ):
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except _EXPECTED_ERRORS as exc:
            await queue.put({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — surface, don't hang the stream
            log.exception("chat_stream_error", session_id=req.session_id)
            await queue.put({"error": f"Internal error: {type(exc).__name__}: {exc}"})
        finally:
            queue.put_nowait(_DONE)

    async def event_stream():
        task = asyncio.create_task(produce())
        loop = asyncio.get_running_loop()
        last_write = loop.time()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=DISCONNECT_POLL_S)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        log.info("client_disconnected", session_id=req.session_id)
                        return
                    if loop.time() - last_write >= KEEPALIVE_S:
                        last_write = loop.time()
                        yield ": keepalive\n\n"
                    continue
                if event is _DONE:
                    break
                last_write = loop.time()
                yield _sse(event)
            yield "data: [DONE]\n\n"
        finally:
            task.cancel()  # no-op when finished; on disconnect, aborts the upstream call

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for live tokens
        },
    )
