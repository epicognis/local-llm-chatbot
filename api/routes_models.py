"""GET /models — curated registry; POST /load — eager model load; POST /unload — purge."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llm.registry import UnknownModelError

log = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models(request: Request) -> list[dict]:
    return request.app.state.registry.public_list()


class LoadRequest(BaseModel):
    model: str                                            # friendly registry name
    num_ctx: int | None = Field(default=None, ge=256)     # default: registry default_ctx
    num_gpu: int | None = Field(default=None, ge=0)       # 0 = load on CPU (Force CPU)
    keep_alive: int | str | None = None                   # None = server default
    purge_others: bool = True                             # unload other resident models first


@router.post("/load")
async def load_model(req: LoadRequest, request: Request) -> dict:
    """Eagerly load a model (and purge others) so it's resident before chatting.

    Always 200 — failures come back as ``status: error`` + a reason so the UI can
    show them without breaking.
    """
    registry = request.app.state.registry
    backend = request.app.state.backend
    try:
        spec = registry.resolve(req.model)
    except UnknownModelError as exc:
        return {"status": "error", "reason": str(exc)}

    keep_alive = req.keep_alive if req.keep_alive is not None else request.app.state.settings.ollama_keep_alive
    try:
        result = await backend.load(
            spec.model, num_ctx=req.num_ctx or spec.default_ctx, num_gpu=req.num_gpu,
            keep_alive=keep_alive, purge_others=req.purge_others,
        )
    except Exception as exc:  # noqa: BLE001 — surface load failures, don't 500
        log.error("model_load_failed", model=spec.model, error=str(exc))
        return {"status": "error", "model": req.model, "tag": spec.model, "reason": str(exc)}
    return {"model": req.model, "tag": spec.model, **result}


@router.post("/unload")
async def unload_all_models(request: Request) -> dict:
    """Unload every resident model to free GPU/VRAM (and RAM) to zero.
    Weights stay on disk; models reload on demand. Always 200."""
    try:
        result = await request.app.state.backend.unload_all()
    except Exception as exc:  # noqa: BLE001
        log.error("model_unload_failed", error=str(exc))
        return {"status": "error", "reason": str(exc)}
    return {"status": "ok", **result}
