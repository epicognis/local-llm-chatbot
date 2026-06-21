import structlog
import structlog.stdlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from session.store import SessionStore
    from orchestrator.orchestrator import Orchestrator

    store = SessionStore(settings.SESSION_DB_URL)
    await store.init()
    app.state.store = store

    if settings.LLM_BACKEND == "ollama":
        from llm.backends.ollama_backend import OllamaBackend
        backend = OllamaBackend(settings.OLLAMA_BASE_URL)
    elif settings.LLM_BACKEND == "mock":
        from llm.backends.mock_backend import MockBackend
        backend = MockBackend()
    elif settings.LLM_BACKEND == "anthropic":
        raise NotImplementedError("Anthropic backend not yet wired — set LLM_BACKEND=ollama or mock")
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND!r}")

    app.state.orchestrator = Orchestrator(backend, store)
    app.state.templates = Jinja2Templates(directory="api/templates")

    log.info(
        "startup",
        backend=settings.LLM_BACKEND,
        default_model=settings.DEFAULT_MODEL,
        session_db=settings.SESSION_DB_URL,
    )

    yield

    log.info("shutdown")


app = FastAPI(title="Local Chatbot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

from api.routes_chat import router as chat_router
from api.routes_models import router as models_router
from api.routes_meta import router as meta_router
from api.routes_sessions import router as sessions_router

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(meta_router)
app.include_router(sessions_router)
