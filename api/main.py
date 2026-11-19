from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import structlog.stdlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import settings

API_DIR = Path(__file__).resolve().parent

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()


def build_backend():
    if settings.LLM_BACKEND == "ollama":
        from llm.backends.ollama_backend import OllamaBackend
        return OllamaBackend(settings.OLLAMA_BASE_URL)
    if settings.LLM_BACKEND == "mock":
        from llm.backends.mock_backend import MockBackend
        return MockBackend()
    raise NotImplementedError(
        f"LLM_BACKEND={settings.LLM_BACKEND!r} is not wired yet — set LLM_BACKEND=ollama or mock"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from orchestrator.orchestrator import Orchestrator
    from session.store import SessionStore

    store = SessionStore(settings.SESSION_DB_URL)
    await store.init()
    app.state.store = store

    backend = build_backend()
    app.state.backend = backend
    app.state.orchestrator = Orchestrator(backend, store)

    log.info(
        "startup",
        backend=settings.LLM_BACKEND,
        default_model=settings.DEFAULT_MODEL,
        context_level=settings.CONTEXT_LEVEL,
        session_db=settings.SESSION_DB_URL,
    )

    yield

    await backend.aclose()
    log.info("shutdown")


app = FastAPI(title="Local Chatbot", lifespan=lifespan)
app.state.templates = Jinja2Templates(directory=API_DIR / "templates")
app.mount("/static", StaticFiles(directory=API_DIR / "static"), name="static")

from api.routes_chat import router as chat_router  # noqa: E402
from api.routes_meta import router as meta_router  # noqa: E402
from api.routes_models import router as models_router  # noqa: E402
from api.routes_sessions import router as sessions_router  # noqa: E402

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(meta_router)
app.include_router(sessions_router)
