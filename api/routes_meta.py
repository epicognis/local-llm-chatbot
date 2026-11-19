from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config.settings import settings

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Liveness. Always 200 while the server is up; Ollama status is informational."""
    result = {"status": "ok", "backend": settings.LLM_BACKEND, "loaded_models": []}
    loaded = getattr(request.app.state.backend, "loaded_models", None)
    if loaded is not None:
        try:
            result["loaded_models"] = await loaded()
            result["ollama"] = "reachable"
        except Exception:
            result["ollama"] = "unreachable"
    return result


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui")


@router.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "ui.html",
        {"default_model": settings.DEFAULT_MODEL},
    )
