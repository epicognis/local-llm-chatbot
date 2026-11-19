import asyncio
import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings
from llm.backends.base import BackendError
from llm.registry import UnknownModelError

router = APIRouter()
log = structlog.get_logger()

# While the model is silent (cold load, summarization, long prefill) send an SSE
# comment this often. It keeps proxies and the UI watchdog happy, and gives the
# server a write on which a vanished client is noticed.
KEEPALIVE_S = 5.0
DISCONNECT_POLL_S = 1.0

_DONE = object()


class ChatOptions(BaseModel):
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, gt=0)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    model: str = Field(default_factory=lambda: settings.DEFAULT_MODEL)
    options: ChatOptions = Field(default_factory=ChatOptions)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _user_error(exc: Exception) -> str:
    if isinstance(exc, (BackendError, UnknownModelError, ValueError)):
        return str(exc)
    return f"Internal error: {type(exc).__name__}: {exc}"


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    orchestrator = request.app.state.orchestrator
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        """Runs the turn in its own task so the response generator can watch for
        disconnects and cancel it — cancelling the task closes the upstream
        Ollama stream (spec §5) and lets the orchestrator persist partial output."""
        try:
            async for event in orchestrator.chat_stream(
                session_id=body.session_id,
                message=body.message,
                model_name=body.model,
                temperature=body.options.temperature,
                max_tokens=body.options.max_tokens,
            ):
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not isinstance(exc, (BackendError, UnknownModelError, ValueError)):
                log.exception("chat_failed", session_id=body.session_id)
            await queue.put({"type": "error", "text": _user_error(exc)})
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
                        log.info("client_disconnected", session_id=body.session_id)
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
            # Normal completion: no-op. Disconnect/cancel: aborts the upstream call.
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
