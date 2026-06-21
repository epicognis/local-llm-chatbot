import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config.settings import settings

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    loaded_models = []
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/ps")
            if resp.status_code == 200:
                data = resp.json()
                loaded_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    return {"status": "ok", "backend": settings.LLM_BACKEND, "loaded_models": loaded_models}


@router.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "ui.html",
        {"default_model": settings.DEFAULT_MODEL},
    )
