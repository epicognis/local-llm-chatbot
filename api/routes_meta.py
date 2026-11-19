"""GET /health (liveness + resident models), GET /ui (the browser UI), GET / -> /ui."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@router.get("/health")
async def health(request: Request) -> dict:
    """Always 200 while the server is up; Ollama status is informational."""
    settings = request.app.state.settings
    backend = request.app.state.backend
    result: dict = {"status": "ok", "backend": settings.llm_backend}
    version = getattr(backend, "version", None)
    if version is not None:
        try:
            result["ollama"] = {
                "reachable": True,
                "version": await version(),
                "loaded": await backend.resident_models(),
            }
        except Exception as exc:  # noqa: BLE001
            result["ollama"] = {"reachable": False, "error": str(exc)}
    return result


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/ui")
async def ui(request: Request):
    # Starlette 1.x: request is the FIRST positional arg, not a context-dict key.
    s = request.app.state.settings
    return _TEMPLATES.TemplateResponse(
        request,
        "ui.html",
        {
            "default_model": s.default_model,
            # Seed values for the LLM Config panel (num_ctx is filled per model client-side).
            "cfg_defaults": {
                "temperature": s.default_temperature,
                "max_tokens": s.default_max_tokens,
                "top_p": s.default_top_p,
                "top_k": s.default_top_k,
            },
        },
        # no-store: never serve a stale cached page over plain-HTTP LAN access.
        headers={"Cache-Control": "no-store"},
    )
