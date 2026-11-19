"""FastAPI app factory + route registration.

Run with uvicorn's factory mode (run.sh does): ``uvicorn api.main:create_app
--factory``. There is deliberately no module-level ``app``: importing this module
must not open the session database (tests import it alongside a live server).

Wires config -> registry -> backend -> session store -> orchestrator and exposes
them on ``app.state`` so routes stay thin. Structured logging via structlog (JSON),
one line per event on stdout (run.sh tees it to server.log for the monitor).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api import routes_chat, routes_meta, routes_models, routes_monitor, routes_sessions
from config.settings import Settings, get_settings
from llm.backends.base import LLMBackend
from llm.backends.mock_backend import MockBackend
from llm.backends.ollama_backend import OllamaBackend
from llm.registry import Registry
from metrics.hw_monitor import HardwareMonitor
from orchestrator.context_budget import ContextBudget
from orchestrator.orchestrator import Orchestrator
from session.store import SessionStore

HERE = Path(__file__).resolve().parent


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs every request at INFO — that would flood server.log with the
    # monitor panel's /api/ps polls.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def build_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "ollama":
        return OllamaBackend(base_url=settings.ollama_base_url)
    return MockBackend()


def create_app(settings: Settings | None = None, backend: LLMBackend | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or get_settings()
    backend = backend or build_backend(settings)
    store = SessionStore(settings.session_db_url)
    log = structlog.get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "app_started",
            backend=settings.llm_backend,
            default_model=settings.default_model,
            context_level=settings.context_level,
            session_db=settings.session_db_url,
        )
        yield
        await backend.aclose()
        store.close()
        log.info("shutdown")

    app = FastAPI(title="Local Chatbot", lifespan=lifespan)

    registry = Registry()
    app.state.settings = settings
    app.state.registry = registry
    app.state.backend = backend
    app.state.store = store
    app.state.orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        backend=backend,
        store=store,
        budget=ContextBudget(
            safety_margin=settings.context_safety_margin,
            level=settings.context_level,
        ),
    )
    app.state.hw_monitor = HardwareMonitor(settings.ollama_base_url)

    app.include_router(routes_meta.router)
    app.include_router(routes_models.router)
    app.include_router(routes_monitor.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_chat.router)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app

